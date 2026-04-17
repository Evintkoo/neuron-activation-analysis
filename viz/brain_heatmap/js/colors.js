// viz/brain_heatmap/js/colors.js
const STOPS = [
    [0.00, 0.019, 0.188, 0.380],
    [0.25, 0.129, 0.400, 0.674],
    [0.50, 0.865, 0.865, 0.865],
    [0.75, 0.705, 0.016, 0.149],
    [1.00, 0.404, 0.000, 0.122],
];

export function coolwarmRgb(t) {
    t = Math.max(0, Math.min(1, t));
    let i = STOPS.findIndex(s => s[0] > t) - 1;
    if (i < 0) i = 0;
    if (i >= STOPS.length - 1) i = STOPS.length - 2;
    const [t0, r0, g0, b0] = STOPS[i];
    const [t1, r1, g1, b1] = STOPS[i + 1];
    const a = (t - t0) / (t1 - t0);
    return [r0 + a * (r1 - r0), g0 + a * (g1 - g0), b0 + a * (b1 - b0)];
}
