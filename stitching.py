'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit.
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py.
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


# small conversion helpers

def img_to_float(t):
    return t.float() / 255.0

def img_to_byte(t):
    return (t.clamp(0.0, 1.0) * 255.0).round().byte()

def to_gray(img):
    return K.color.rgb_to_grayscale(img_to_float(img).unsqueeze(0))

def identity(dev):
    return torch.eye(3, dtype=torch.float32, device=dev)

def make_translation(tx, ty, dev):
    T = identity(dev)
    T[0, 2] = tx
    T[1, 2] = ty
    return T

def image_corners(h, w, dev):
    return torch.tensor(
        [[0., 0.], [w-1., 0.], [w-1., h-1.], [0., h-1.]],
        dtype=torch.float32, device=dev)


def safe_downsample(img, max_side=1000):
    # downsample large images to speed up matching
    _, h, w = img.shape
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img, 1.0
    nh = max(64, int(h * scale))
    nw = max(64, int(w * scale))
    x = img_to_float(img).unsqueeze(0)
    small = torch.nn.functional.interpolate(
        x, size=(nh, nw), mode='bilinear', align_corners=False)[0]
    return img_to_byte(small), scale


def cylindrical_warp_image(img, focal=None):
    # cylindrical warp for panorama stability
    x = img_to_float(img).unsqueeze(0)
    _, _, h, w = x.shape
    dev = img.device
    if focal is None:
        focal = 0.85 * float(w)
    ys = torch.linspace(0, h - 1, h, device=dev)
    xs = torch.linspace(0, w - 1, w, device=dev)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    theta = (xx - cx) / focal
    y_hat = (yy - cy) / focal

    X = torch.tan(theta)
    Y = y_hat * torch.sqrt(1.0 + X * X)

    src_x = focal * X + cx
    src_y = focal * Y + cy

    gx = (src_x / max(w - 1, 1)) * 2.0 - 1.0
    gy = (src_y / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

    warped = torch.nn.functional.grid_sample(
        x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    ones = torch.ones(1, 1, h, w, dtype=torch.float32, device=dev)
    mask = torch.nn.functional.grid_sample(
        ones, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    mask = (mask > 0.5).float()

    valid_cols = torch.where(mask[0, 0].sum(0) > 0)[0]
    if valid_cols.numel() > 0:
        l = int(valid_cols.min())
        r = int(valid_cols.max()) + 1
        warped = warped[:, :, :, l:r]
    return img_to_byte(warped[0])


def largest_valid_rect(mask):
    # crop to a strong interior valid region
    m = mask[0, 0] > 0.6
    h, w = m.shape
    if h == 0 or w == 0:
        return 0, 0, 0, 0

    heights = torch.zeros(w, dtype=torch.int64, device=mask.device)
    best_area = 0
    best = (0, h - 1, 0, w - 1)

    for y in range(h):
        heights = torch.where(m[y], heights + 1, torch.zeros_like(heights))
        stack = []
        x = 0
        while x <= w:
            cur = int(heights[x].item()) if x < w else 0
            if not stack or cur >= stack[-1][1]:
                stack.append((x, cur))
                x += 1
                continue

            idx, hh = stack.pop()
            left = stack[-1][0] + 1 if stack else 0
            width = x - left
            area = hh * width
            if area > best_area and hh > 0 and width > 0:
                y1 = y - hh + 1
                y2 = y
                x1 = left
                x2 = x - 1
                best_area = area
                best = (y1, y2, x1, x2)
    return best

def apply_homography(H, pts):
    n = pts.shape[0]
    ones = torch.ones(n, 1, dtype=pts.dtype, device=pts.device)
    ph = torch.cat([pts, ones], dim=1)
    out = (H @ ph.t()).t()
    z = out[:, 2:3]
    z = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
    return out[:, :2] / z


def warp_with_mask(img, H, out_h, out_w):
    x = img_to_float(img).unsqueeze(0)
    _, _, h, w = x.shape
    warped = K.geometry.transform.warp_perspective(
        x, H.unsqueeze(0), dsize=(out_h, out_w), align_corners=True)
    ones = torch.ones(1, 1, h, w, dtype=torch.float32, device=img.device)
    wmask = K.geometry.transform.warp_perspective(
        ones, H.unsqueeze(0), dsize=(out_h, out_w), align_corners=True)
    wmask = (wmask > 0.5).float()
    return warped, wmask


def compute_canvas(H_map, sizes, dev):
    all_pts = []
    for idx, H in H_map.items():
        h, w = sizes[idx]
        pts = apply_homography(H, image_corners(h, w, dev))
        if torch.isfinite(pts).all():
            all_pts.append(pts)
    if not all_pts:
        return identity(dev), 64, 64
    pts = torch.clamp(torch.cat(all_pts, 0), -30000., 30000.)
    mn = torch.floor(pts.min(0).values)
    mx = torch.ceil(pts.max(0).values)
    tx = float(-mn[0].item()) if mn[0] < 0 else 0.
    ty = float(-mn[1].item()) if mn[1] < 0 else 0.
    ow = max(32, min(int(mx[0].item() + tx + 1), 16000))
    oh = max(32, min(int(mx[1].item() + ty + 1), 10000))
    return make_translation(tx, ty, dev), oh, ow


# ----
# Harris + patch descriptor feature pipeline
# ----

def harris_response(gray, k=0.04):
    grad = K.filters.spatial_gradient(gray, order=1, normalized=False)
    ix, iy = grad[:, :, 0], grad[:, :, 1]
    ix2 = K.filters.gaussian_blur2d(ix*ix, (5,5), (1.2,1.2))
    iy2 = K.filters.gaussian_blur2d(iy*iy, (5,5), (1.2,1.2))
    ixy = K.filters.gaussian_blur2d(ix*iy, (5,5), (1.2,1.2))
    det = ix2*iy2 - ixy**2
    return det - k*(ix2+iy2)**2


def pick_keypoints(gray, max_pts=12000, nms_r=5, border=12):
    R = harris_response(gray).clone()
    _, _, h, w = R.shape
    R[:,:,:border,:] = 0
    R[:,:,h-border:,:] = 0
    R[:,:,:,:border] = 0
    R[:,:,:,w-border:] = 0
    pooled = torch.nn.functional.max_pool2d(R, nms_r, stride=1, padding=nms_r//2)
    keep = (R == pooled) & (R > 1e-7)
    ys, xs = torch.where(keep[0, 0])
    if ys.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=gray.device)
    scores = R[0, 0, ys, xs]
    k = min(max_pts, scores.numel())
    _, top = torch.topk(scores, k)
    return torch.stack([xs[top].float(), ys[top].float()], dim=1)


def patch_descriptors(gray, pts, patch=21):
    dev = gray.device
    if pts.shape[0] == 0:
        return torch.empty((0, patch*patch), dtype=torch.float32, device=dev)
    g = K.filters.gaussian_blur2d(gray, (5,5), (1.,1.))
    pad = patch // 2
    padded = torch.nn.functional.pad(g, (pad,pad,pad,pad), mode='reflect')
    _, _, h, w = g.shape
    unf = torch.nn.functional.unfold(padded, kernel_size=(patch,patch), stride=1)
    xs = pts[:,0].round().long().clamp(0, w-1)
    ys = pts[:,1].round().long().clamp(0, h-1)
    idx = ys*w + xs
    d = unf[0,:,idx].t().contiguous()
    d = d - d.mean(1, keepdim=True)
    d = d / (d.std(1, keepdim=True, unbiased=False) + 1e-6)
    return torch.nn.functional.normalize(d, p=2, dim=1)


def match_descriptors(d1, d2, ratio=0.90):
    if d1.shape[0] < 4 or d2.shape[0] < 4:
        empty = torch.empty(0, dtype=torch.long, device=d1.device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=d1.device)
    dists = torch.cdist(d1, d2)
    vals, inds = torch.topk(dists, k=2, largest=False, dim=1)
    best = inds[:, 0]
    ratio_ok = vals[:,0] / (vals[:,1] + 1e-8) < ratio
    nn21 = torch.argmin(dists, dim=0)
    mutual = nn21[best] == torch.arange(d1.shape[0], device=d1.device)
    keep = ratio_ok & mutual
    ii = torch.arange(d1.shape[0], device=d1.device)[keep]
    jj = best[keep]
    conf = 1.0 / (vals[:,0][keep] + 1e-6)
    if conf.numel() > 0:
        order = torch.argsort(conf, descending=True)
        ii, jj, conf = ii[order], jj[order], conf[order]
    return ii, jj, conf


# ----
# Homography estimation: DLT + RANSAC
# ----

def normalise_pts(pts):
    mean = pts.mean(0)
    shifted = pts - mean
    avg_dist = torch.norm(shifted, dim=1).mean() + 1e-8
    scale = (2.0**0.5) / avg_dist.double()
    T = torch.zeros(3, 3, dtype=torch.float64, device=pts.device)
    T[0,0]=scale; T[1,1]=scale
    T[0,2]=-scale*mean[0].double()
    T[1,2]=-scale*mean[1].double()
    T[2,2]=1.
    ones = torch.ones(pts.shape[0], 1, dtype=torch.float64, device=pts.device)
    ph = torch.cat([pts.double(), ones], 1)
    return (T @ ph.t()).t()[:,:2], T


def dlt_homography(src, dst):
    if src.shape[0] < 4:
        return None
    sn, Ts = normalise_pts(src)
    dn, Td = normalise_pts(dst)
    n = src.shape[0]
    A = torch.zeros(2*n, 9, dtype=torch.float64, device=src.device)
    x,y = sn[:,0], sn[:,1]
    u,v = dn[:,0], dn[:,1]
    A[0::2,0]=-x; A[0::2,1]=-y; A[0::2,2]=-1
    A[0::2,6]=u*x; A[0::2,7]=u*y; A[0::2,8]=u
    A[1::2,3]=-x; A[1::2,4]=-y; A[1::2,5]=-1
    A[1::2,6]=v*x; A[1::2,7]=v*y; A[1::2,8]=v
    try:
        _, _, vh = torch.linalg.svd(A)
    except RuntimeError:
        return None
    Hn = vh[-1].reshape(3,3)
    H = torch.linalg.inv(Td) @ Hn @ Ts
    if H[2,2].abs() < 1e-12:
        return None
    return (H / H[2,2]).float()


def affine_from_pts(src, dst):
    n = src.shape[0]
    if n < 3:
        return None
    dev = src.device
    A = torch.zeros(2*n, 6, dtype=torch.float32, device=dev)
    b = torch.zeros(2*n, dtype=torch.float32, device=dev)
    x,y = src[:,0], src[:,1]
    A[0::2,0]=x; A[0::2,1]=y; A[0::2,2]=1.
    A[1::2,3]=x; A[1::2,4]=y; A[1::2,5]=1.
    b[0::2]=dst[:,0]; b[1::2]=dst[:,1]
    try:
        sol = torch.linalg.lstsq(A, b).solution
    except RuntimeError:
        return None
    H = identity(dev)
    H[0,0]=sol[0]; H[0,1]=sol[1]; H[0,2]=sol[2]
    H[1,0]=sol[3]; H[1,1]=sol[4]; H[1,2]=sol[5]
    return H


def ransac_homography(src, dst, iters=7000, thr=5.0):
    n = src.shape[0]
    if n < 4:
        return None, None
    best_H, best_mask, best_n = None, None, 0
    for _ in range(iters):
        idx = torch.randperm(n, device=src.device)[:4]
        H = dlt_homography(src[idx], dst[idx])
        if H is None:
            continue
        err = torch.norm(apply_homography(H, src) - dst, dim=1)
        mask = err < thr
        cnt = int(mask.sum())
        if cnt > best_n:
            best_n, best_mask, best_H = cnt, mask, H
    if best_H is None or best_n < 4:
        return None, None
    inl = torch.where(best_mask)[0]
    if inl.numel() > 400:
        inl = inl[torch.randperm(inl.numel(), device=src.device)[:400]]
    H2 = dlt_homography(src[inl], dst[inl])
    if H2 is None:
        H2 = best_H
    err2 = torch.norm(apply_homography(H2, src) - dst, dim=1)
    return H2, err2 < thr


def ransac_affine(src, dst, iters=6000, thr=4.5):
    n = src.shape[0]
    if n < 3:
        return None, None
    best_H, best_mask, best_n = None, None, 0
    for _ in range(iters):
        idx = torch.randperm(n, device=src.device)[:3]
        H = affine_from_pts(src[idx], dst[idx])
        if H is None:
            continue
        err = torch.norm(apply_homography(H, src) - dst, dim=1)
        mask = err < thr
        cnt = int(mask.sum())
        if cnt > best_n:
            best_n, best_mask, best_H = cnt, mask, H
    if best_H is None or best_n < 3:
        return None, None
    inl = torch.where(best_mask)[0]
    if inl.numel() > 400:
        inl = inl[torch.randperm(inl.numel(), device=src.device)[:400]]
    H2 = affine_from_pts(src[inl], dst[inl])
    if H2 is None:
        H2 = best_H
    err2 = torch.norm(apply_homography(H2, src) - dst, dim=1)
    return H2, err2 < thr


def estimate_transform(ref, src, model='homography',
                        max_kp=4000, ratio=0.85,
                        ransac_iters=3000, ransac_thr=5.0):
    ref_s, sc_r = safe_downsample(ref, max_side=1000)
    src_s, sc_s = safe_downsample(src, max_side=1000)

    gr = to_gray(ref_s)
    gs = to_gray(src_s)
    pr = pick_keypoints(gr, max_kp)
    ps = pick_keypoints(gs, max_kp)
    dr = patch_descriptors(gr, pr)
    ds = patch_descriptors(gs, ps)
    ii, jj, _ = match_descriptors(dr, ds, ratio)
    if ii.numel() < 4:
        return None, 0, 0
    mr, ms = pr[ii], ps[jj]

    mr_orig = mr.clone()
    ms_orig = ms.clone()
    mr_orig[:, 0] /= sc_r;  mr_orig[:, 1] /= sc_r
    ms_orig[:, 0] /= sc_s;  ms_orig[:, 1] /= sc_s

    if model == 'affine':
        H, inl = ransac_affine(ms_orig, mr_orig, ransac_iters, ransac_thr)
    else:
        H, inl = ransac_homography(ms_orig, mr_orig, ransac_iters, ransac_thr)
    if H is None:
        return None, int(ii.numel()), 0
    return H, int(ii.numel()), int(inl.sum())


def valid_transform(H, h, w):
    if H is None or not torch.isfinite(H).all():
        return False
    det = H[0,0]*H[1,1] - H[0,1]*H[1,0]
    if not (0.003 < det.abs().item() < 1000.):
        return False
    pts = apply_homography(H, image_corners(h, w, H.device))
    if not torch.isfinite(pts).all():
        return False
    span = pts.max(0).values - pts.min(0).values
    return not (span[0] > 40000. or span[1] > 40000.)


def bbox_overlap(H_j_to_i, si, sj, dev):
    hi, wi = si; hj, wj = sj
    ci = image_corners(hi, wi, dev)
    cj = apply_homography(H_j_to_i, image_corners(hj, wj, dev))
    mni, mxi = ci.min(0).values, ci.max(0).values
    mnj, mxj = cj.min(0).values, cj.max(0).values
    iw = (torch.min(mxi[0],mxj[0]) - torch.max(mni[0],mnj[0])).clamp(0.)
    ih = (torch.min(mxi[1],mxj[1]) - torch.max(mni[1],mnj[1])).clamp(0.)
    inter = iw * ih
    ai = (mxi[0]-mni[0]).clamp(1.) * (mxi[1]-mni[1]).clamp(1.)
    aj = (mxj[0]-mnj[0]).clamp(1.) * (mxj[1]-mnj[1]).clamp(1.)
    return float((inter / torch.min(ai, aj)).item())


# ----
# Task 1: background reconstruction / foreground removal
# ----

def estimate_background(img, valid_mask, overlap_mask, kernel=101):
    # estimate background from non-overlap regions
    non_ov = valid_mask * (1.0 - overlap_mask)
    k = kernel if kernel % 2 == 1 else kernel + 1
    sig = k / 4.0
    num = K.filters.gaussian_blur2d(img * non_ov, (k,k), (sig,sig))
    den = K.filters.gaussian_blur2d(non_ov, (k,k), (sig,sig)).clamp(1e-6)
    return num / den


def remove_foreground(w1, m1, w2, m2):
    ov    = m1 * m2
    union = ((m1 + m2) > 0).float()

    if float(ov.sum()) < 1.0:
        # no overlap, just paste
        return w1*m1 + w2*m2*(1.0-m1), union

    # detect disagreement --- likely foreground
    diff   = torch.abs(w1 - w2).mean(1, keepdim=True)
    diff_s = K.filters.gaussian_blur2d(diff * ov, (15,15), (4.,4.))

    sample = diff_s[ov > 0.5]
    if sample.numel() > 10:
        thr = max(float(torch.quantile(sample, 0.15)), 0.015)
    else:
        thr = 0.015

    moving = (diff_s > thr).float() * ov

    # expand the mask to catch motion blur around the object
    moving = K.morphology.dilation(
        moving, torch.ones(55, 55, device=w1.device, dtype=w1.dtype)) * ov

    # soften the mask before blending
    moving_s = K.filters.gaussian_blur2d(moving, (41,41), (11.,11.)) * ov

    # estimate smooth background using large kernel
    bg1 = estimate_background(w1, m1, ov, kernel=201)
    bg2 = estimate_background(w2, m2, ov, kernel=201)

    dev1 = K.filters.gaussian_blur2d(
        torch.abs(w1-bg1).mean(1, keepdim=True)*ov, (41,41),(11.,11.))
    dev2 = K.filters.gaussian_blur2d(
        torch.abs(w2-bg2).mean(1, keepdim=True)*ov, (41,41),(11.,11.))

    # prefer the image that looks more background-like
    static_zone = ov * (1.0 - moving)

    static_sim1 = K.filters.gaussian_blur2d(
        (1.0 - torch.abs(w1 - w2).mean(1,keepdim=True)) * static_zone,
        (41,41),(11.,11.))
    static_sim2 = K.filters.gaussian_blur2d(
        (1.0 - torch.abs(w2 - w1).mean(1,keepdim=True)) * static_zone,
        (41,41),(11.,11.))

    score1 = dev1 - 0.35 * static_sim1
    score2 = dev2 - 0.35 * static_sim2

    sel1 = (score1 <= score2).float()
    sel2 = 1.0 - sel1

    static_ov = ov * (1.0 - moving_s)
    moving_ov = ov * moving_s

    bw1 = m1*(1.0-ov) + 0.5*static_ov + sel1*moving_ov
    bw2 = m2*(1.0-ov) + 0.5*static_ov + sel2*moving_ov
    denom = (bw1 + bw2).clamp(1e-6)
    result = (w1*bw1 + w2*bw2) / denom

    # final cleanup for strong ghosting regions
    strong_moving = (moving_s > 0.6) * ov
    if float(strong_moving.sum()) > 0:
        diff1_bg = torch.abs(w1 - bg1).mean(1, keepdim=True)
        diff2_bg = torch.abs(w2 - bg2).mean(1, keepdim=True)
        use1 = (diff1_bg <= diff2_bg).float()
        clean = use1 * w1 + (1.0 - use1) * w2
        result = torch.where(strong_moving.expand_as(result) > 0.5, clean, result)

    return result, union


# ----
# Graph construction and spanning tree for panorama
# ----

def score_pair(images, i, j, dev, model='homography'):
    _, hi, wi = images[i].shape
    _, hj, wj = images[j].shape

    H, rh, ih = estimate_transform(images[i], images[j], model,
                                   4000, 0.84 if model == 'affine' else 0.85,
                                   3500, 4.0 if model == 'affine' else 5.0)
    if H is None:
        return None, 0., 0

    min_inliers = 10 if model == 'affine' else 8
    if ih < min_inliers or not valid_transform(H, hj, wj):
        return None, 0., 0

    ov = bbox_overlap(H, (hi, wi), (hj, wj), dev)
    if ov < (0.12 if model == 'affine' else 0.15):
        return None, 0., 0

    sc = float(ih) + 0.01 * float(rh) + 8.0 * ov
    return H, sc, ih


def build_graph(images, dev, model='homography'):
    n = len(images)
    ov = torch.eye(n, dtype=torch.int64, device=dev)
    pair_H = {}
    scores = torch.zeros(n, n, dtype=torch.float32, device=dev)
    for i in range(n):
        for j in range(i + 1, n):
            H, sc, inl = score_pair(images, i, j, dev, model)
            if H is None:
                continue
            ov[i, j] = ov[j, i] = 1
            scores[i, j] = scores[j, i] = sc
            pair_H[(j, i)] = H
            try:
                pair_H[(i, j)] = torch.linalg.inv(H)
            except RuntimeError:
                pair_H[(i, j)] = identity(dev)
    return ov, pair_H, scores


def max_spanning_tree(scores):
    n = scores.shape[0]
    used = torch.zeros(n, dtype=torch.bool, device=scores.device)
    used[0] = True
    edges = []
    for _ in range(n-1):
        bs, bu, bv = -1., -1, -1
        for u in range(n):
            if not used[u]:
                continue
            for v in range(n):
                if used[v]:
                    continue
                s = float(scores[u,v])
                if s > bs:
                    bs, bu, bv = s, u, v
        if bu < 0 or bs <= 0:
            break
        used[bv] = True
        edges.append((bu, bv))
    return edges


def bfs_order(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    connected = [i for i in range(n) if adj[i]]
    if not connected:
        return [0]
    leaves = [i for i in connected if len(adj[i]) == 1]
    start = leaves[0] if leaves else connected[0]
    order, seen, q = [], set(), [start]
    while q:
        cur = q.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        order.append(cur)
        q.extend(nb for nb in adj[cur] if nb not in seen)
    return order


def compose_transforms(order, pair_H, dev):
    if not order:
        return {}, 0
    ref = order[len(order)//2]
    H_map = {ref: identity(dev)}
    nbrs = {}
    for (a,b) in pair_H:
        nbrs.setdefault(a, [])
        if b not in nbrs[a]:
            nbrs[a].append(b)
    visited, q = {ref}, [ref]
    while q:
        cur = q.pop(0)
        for nb in nbrs.get(cur, []):
            if nb in visited or nb not in order:
                continue
            if (nb, cur) in pair_H:
                H_map[nb] = H_map[cur] @ pair_H[(nb, cur)]
                visited.add(nb)
                q.append(nb)
    return H_map, ref


def level_panorama(H_map, order, sizes, dev):
    keys = [i for i in order if i in H_map]
    if len(keys) < 2:
        return H_map
    all_corners = []
    for i in keys:
        h, w = sizes[i]
        pts = apply_homography(H_map[i], image_corners(h, w, dev))
        if torch.isfinite(pts).all():
            all_corners.append(pts)
    if not all_corners:
        return H_map
    cat = torch.cat(all_corners, 0)
    cx, cy = cat[:,0].mean(), cat[:,1].mean()

    centers = []
    for i in keys:
        h, w = sizes[i]
        c = apply_homography(H_map[i], image_corners(h,w,dev)).mean(0)
        centers.append(c)
    centers = torch.stack(centers)
    xs, ys = centers[:,0], centers[:,1]

    denom = ((xs-xs.mean())**2).sum()
    if float(denom) <= 1e-6:
        return H_map

    slope = ((xs-xs.mean())*(ys-ys.mean())).sum() / denom
    angle = torch.atan(slope).clamp(-12*3.14159/180., 12*3.14159/180.)
    ca, sa = torch.cos(-angle), torch.sin(-angle)

    R = identity(dev)
    R[0,0]=ca; R[0,1]=-sa; R[0,2]=cx*(1-ca)+cy*sa
    R[1,0]=sa; R[1,1]= ca; R[1,2]=cy*(1-ca)-cx*sa

    rotated = {i: R@H for i, H in H_map.items()}

    all_pts = []
    for i, H in rotated.items():
        h, w = sizes[i]
        pts = apply_homography(H, image_corners(h, w, dev))
        if torch.isfinite(pts).all():
            all_pts.append(pts)
    if not all_pts:
        return rotated
    cat2 = torch.cat(all_pts, 0)
    mn = cat2.min(0).values
    shift_x = float(-mn[0].item()) if mn[0] < 0 else 0.
    shift_y = float(-mn[1].item()) if mn[1] < 0 else 0.
    if shift_x == 0. and shift_y == 0.:
        return rotated
    S = make_translation(shift_x, shift_y, dev)
    return {i: S@H for i, H in rotated.items()}


def overlap_matrix(H_map, sizes, n, dev):
    result = torch.eye(n, dtype=torch.int64, device=dev)
    polys = {}
    for i, H in H_map.items():
        h, w = sizes[i]
        polys[i] = apply_homography(H, image_corners(h, w, dev))
    for i in polys:
        mni, mxi = polys[i].min(0).values, polys[i].max(0).values
        ai = (mxi[0]-mni[0]).clamp(1.) * (mxi[1]-mni[1]).clamp(1.)
        for j in polys:
            if i == j:
                continue
            mnj, mxj = polys[j].min(0).values, polys[j].max(0).values
            aj = (mxj[0]-mnj[0]).clamp(1.) * (mxj[1]-mnj[1]).clamp(1.)
            iw = (torch.min(mxi[0],mxj[0]) - torch.max(mni[0],mnj[0])).clamp(0.)
            ih = (torch.min(mxi[1],mxj[1]) - torch.max(mni[1],mnj[1])).clamp(0.)
            if iw*ih > 0.20*torch.min(ai,aj):
                result[i,j] = result[j,i] = 1
    return result


# ----
# Distance-to-edge weight map and panorama rendering
# ----

def distance_weight_map(h, w, dev):
    y = torch.arange(h, dtype=torch.float32, device=dev)
    x = torch.arange(w, dtype=torch.float32, device=dev)
    dy = 1.0 - ((y-(h-1)*0.5)/(0.5*max(h-1,1))).abs()
    dx = 1.0 - ((x-(w-1)*0.5)/(0.5*max(w-1,1))).abs()
    wmap = (dy.view(h,1)*dx.view(1,w)).clamp(0.)
    return (wmap**2.2).unsqueeze(0).unsqueeze(0) + 1e-6


def warp_weight(h, w, H, oh, ow, mask):
    base = distance_weight_map(h, w, H.device)
    ww = K.geometry.transform.warp_perspective(
        base, H.unsqueeze(0), dsize=(oh,ow), align_corners=True)
    return ww * mask


def render_panorama(images, H_map, active, dev):
    sizes = {i: (images[i].shape[1], images[i].shape[2]) for i in active}
    T, oh, ow = compute_canvas(H_map, sizes, dev)

    if oh < 10 or ow < 10:
        return img_to_byte(img_to_float(images[active[0]]).unsqueeze(0))[0]

    warped, masks, weights = {}, {}, {}
    for i in active:
        _, h, w = images[i].shape
        H = T @ H_map[i]
        wi, mi = warp_with_mask(images[i], H, oh, ow)
        if float(mi.sum()) < 100:
            continue
        wt = warp_weight(h, w, H, oh, ow, mi)
        warped[i] = wi
        masks[i] = mi
        weights[i] = wt

    if not warped:
        return img_to_byte(img_to_float(images[active[0]]).unsqueeze(0))[0]

    num = torch.zeros(1, 3, oh, ow, dtype=torch.float32, device=dev)
    den = torch.zeros(1, 1, oh, ow, dtype=torch.float32, device=dev)
    for i in warped:
        num += warped[i] * weights[i]
        den += weights[i]

    covered = den > 1e-6
    result = torch.where(covered, num / den.clamp(1e-6), torch.zeros_like(num))

    cov = covered.float()
    fill = result.clone()
    for _ in range(4):
        ecov = torch.nn.functional.max_pool2d(cov, 3, stride=1, padding=1)
        en = K.filters.gaussian_blur2d(fill * cov, (5, 5), (1.5, 1.5))
        ed = K.filters.gaussian_blur2d(cov, (5, 5), (1.5, 1.5)).clamp(1e-6)
        diff = (ecov > 0.5) & (cov < 0.5)
        fill = torch.where(diff.expand_as(fill), en / ed, fill)
        cov = torch.where(diff, ecov, cov)

    y1, y2, x1, x2 = largest_valid_rect(cov)
    if y2 > y1 and x2 > x1:
        fill = fill[:, :, int(y1):int(y2) + 1, int(x1):int(x2) + 1]
    else:
        ys, xs = torch.where(cov[0, 0] > 0.5)
        if ys.numel() > 0:
            fill = fill[:, :, int(ys.min()):int(ys.max()) + 1,
                            int(xs.min()):int(xs.max()) + 1]
    return img_to_byte(fill[0])


# ----
# Task 1
# ----

def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    keys = sorted(imgs.keys())
    img1 = imgs[keys[0]].float()
    img2 = imgs[keys[1]].float()
    dev  = img1.device

    _, h1, w1 = img1.shape
    _, h2, w2 = img2.shape

    H, _, _ = estimate_transform(img1, img2, 'homography', 6000, 0.90, 5000, 4.0)
    if H is None:
        oh = max(h1, h2)
        out = torch.zeros(3, oh, w1+w2, dtype=torch.uint8, device=dev)
        out[:,:h1,:w1] = img1.byte()
        out[:,:h2,w1:] = img2.byte()
        return out

    H_map = {0: identity(dev), 1: H}
    sizes  = {0:(h1,w1), 1:(h2,w2)}
    T, oh, ow = compute_canvas(H_map, sizes, dev)

    w1_img, w1_mask = warp_with_mask(img1, T, oh, ow)
    w2_img, w2_mask = warp_with_mask(img2, T@H, oh, ow)

    blended, _ = remove_foreground(w1_img, w1_mask, w2_img, w2_mask)
    return img_to_byte(blended[0])


# ----
# Task 2 + Bonus 1 + Bonus 2
# ----

def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama,
        overlap: torch.Tensor of the output image.
    """
    keys = sorted(imgs.keys())
    raw_images = [imgs[k].float() for k in keys]
    dev = raw_images[0].device
    n = len(raw_images)

    pano_images = []
    for im in raw_images:
        _, h, w = im.shape
        use_cyl = (n >= 4) or (h > w)
        pano_images.append(cylindrical_warp_image(im).float() if use_cyl else im)

    model = 'affine' if n >= 4 else 'homography'
    _, pair_H, scores = build_graph(pano_images, dev, model)
    edges = max_spanning_tree(scores)

    connected = set()
    for u, v in edges:
        connected.add(u)
        connected.add(v)
    if not connected:
        return pano_images[0].byte(), torch.eye(n, dtype=torch.int64, device=dev)

    order = bfs_order(n, edges)
    H_map, _ = compose_transforms(order, pair_H, dev)

    active = [i for i in order if i in H_map]
    if not active:
        return pano_images[0].byte(), torch.eye(n, dtype=torch.int64, device=dev)

    sizes = {i: (pano_images[i].shape[1], pano_images[i].shape[2]) for i in active}
    H_map = level_panorama(H_map, order, sizes, dev)
    ov = overlap_matrix(H_map, sizes, n, dev)
    pano = render_panorama(pano_images, H_map, active, dev)

    return pano, ov