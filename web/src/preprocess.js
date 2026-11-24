// Image preprocessing that reproduces the Python pipeline exactly:
//   PIL Image.resize((224, 224), BILINEAR)  ->  ToTensor  ->  Normalize(ImageNet)
// The browser's canvas resize uses a different filter, so we port PIL's
// two-pass, fixed-point, antialiased bilinear resampler (libImaging/Resample.c).
// eval/check_web_preprocess.py verifies this matches Python pixel-for-pixel.

export const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const PRECISION_BITS = 32 - 8 - 2;

function bilinearFilter(x) {
  x = Math.abs(x);
  return x < 1 ? 1 - x : 0;
}

// Precompute fixed-point coefficients for resampling `inSize` -> `outSize` (PIL precompute_coeffs).
function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 1 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const tmp = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    // PIL casts with (int), which truncates toward zero.
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const w = bilinearFilter((x + xmin - center + 0.5) * ss);
      tmp[x] = w;
      ww += w;
    }
    for (let x = 0; x < ksize; x++) {
      const k = x < xmax && ww !== 0 ? tmp[x] / ww : 0;
      // PIL normalize_coeffs_8bpc: round half away from zero
      kk[xx * ksize + x] = k < 0 ? Math.trunc(-0.5 + k * (1 << PRECISION_BITS))
                                 : Math.trunc(0.5 + k * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, kk, ksize };
}

// PIL accumulates in a 32-bit int and shifts right; JS bit ops are 32-bit too but
// overflow differently, so accumulate in float (exact for integers < 2^53) and floor.
function clip8f(acc) {
  const r = Math.floor(acc / (1 << PRECISION_BITS));
  return r < 0 ? 0 : r > 255 ? 255 : r;
}

// rgb: Uint8ClampedArray / Uint8Array with 3 (RGB) or 4 (RGBA) channels per pixel.
export function resizePIL(rgb, w, h, outW, outH, channels = 4) {
  const init = 1 << (PRECISION_BITS - 1);
  // Horizontal pass first, then vertical, rounding to 8 bits in between (as PIL does).
  const H = coeffs(w, outW);
  const tmp = new Uint8Array(outW * h * 3);
  for (let y = 0; y < h; y++) {
    for (let xx = 0; xx < outW; xx++) {
      const xmin = H.bounds[xx * 2], xmax = H.bounds[xx * 2 + 1];
      const k = xx * H.ksize;
      let r = init, g = init, b = init;
      for (let x = 0; x < xmax; x++) {
        const c = H.kk[k + x];
        const p = (y * w + x + xmin) * channels;
        r += rgb[p] * c; g += rgb[p + 1] * c; b += rgb[p + 2] * c;
      }
      const o = (y * outW + xx) * 3;
      tmp[o] = clip8f(r); tmp[o + 1] = clip8f(g); tmp[o + 2] = clip8f(b);
    }
  }
  // Vertical pass
  const V = coeffs(h, outH);
  const out = new Uint8Array(outW * outH * 3);
  for (let yy = 0; yy < outH; yy++) {
    const ymin = V.bounds[yy * 2], ymax = V.bounds[yy * 2 + 1];
    const k = yy * V.ksize;
    for (let xx = 0; xx < outW; xx++) {
      let r = init, g = init, b = init;
      for (let y = 0; y < ymax; y++) {
        const c = V.kk[k + y];
        const p = ((y + ymin) * outW + xx) * 3;
        r += tmp[p] * c; g += tmp[p + 1] * c; b += tmp[p + 2] * c;
      }
      const o = (yy * outW + xx) * 3;
      out[o] = clip8f(r); out[o + 1] = clip8f(g); out[o + 2] = clip8f(b);
    }
  }
  return out;
}

// RGB uint8 (224x224x3, HWC) -> normalised Float32Array (1x3x224x224, CHW)
export function toTensor(rgb224) {
  const n = SIZE * SIZE;
  const t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      t[c * n + i] = (rgb224[i * 3 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return t;
}
