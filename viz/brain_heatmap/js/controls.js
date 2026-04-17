// viz/brain_heatmap/js/controls.js
const LAYER_NAMES = ['early', 'mid', 'late'];
const LAYER_LABELS = ['Early (low-level)', 'Mid (semantic)', 'Late (output)'];

export function initControls(setType, setLayer) {
    document.getElementById('type-select').addEventListener('change', e => {
        setType(e.target.value);
    });

    const slider = document.getElementById('layer-slider');
    const layerLabel = document.getElementById('layer-label');
    slider.addEventListener('input', e => {
        const idx = parseInt(e.target.value, 10);
        layerLabel.textContent = LAYER_LABELS[idx];
        setLayer(LAYER_NAMES[idx]);
    });
}
