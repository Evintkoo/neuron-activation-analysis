// viz/brain_heatmap/js/app.js
import { init, setType, setLayer } from './brain.js';
import { initControls } from './controls.js';
import { initOverlay } from './overlay.js';

fetch('data/viz_data.json')
    .then(r => r.json())
    .then(data => {
        init(data);
        initControls(setType, setLayer);
        initOverlay(data);
    })
    .catch(err => {
        console.error('Failed to load viz data:', err);
        document.getElementById('status').textContent = 'Error: ' + err.message;
    });
