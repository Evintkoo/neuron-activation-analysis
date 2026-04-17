// viz/brain_heatmap/js/dashboard.js
import { coolwarmRgb } from './colors.js';

fetch('data/viz_data.json')
    .then(r => r.json())
    .then(data => {
        renderTheoryChart(data.theory_fit);
        renderStats(data);
        renderContrastiveDelta(data.contrastive_deltas);
        initExplorer();
    })
    .catch(err => {
        console.error('Failed to load viz data:', err);
    });

function renderTheoryChart(fit) {
    if (!fit) return;
    const { dct_score, gwt_score, fep_score, iit_score } = fit;
    const labels = ['DCT', 'GWT', 'FEP', 'IIT'];
    const scores = [dct_score, gwt_score, fep_score, iit_score];
    if (scores.some(s => typeof s !== 'number' || isNaN(s))) {
        console.error('Invalid theory_fit data: non-numeric scores');
        return;
    }
    const maxIdx = scores.indexOf(Math.max(...scores));
    const bgColors = scores.map((_, i) =>
        i === maxIdx ? 'rgba(240,192,64,0.85)' : 'rgba(124,156,191,0.6)'
    );
    new Chart(document.getElementById('theory-chart'), {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Theory Fit', data: scores,
            backgroundColor: bgColors, borderWidth: 1 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 1,
                     ticks: { color: '#aaa' }, grid: { color: '#1e1e3a' } },
                x: { ticks: { color: '#e0e0e0' }, grid: { color: 'transparent' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderStats(data) {
    if (!data?.theory_fit) return;
    const { dct_score, gwt_score, fep_score, iit_score } = data.theory_fit;
    const allScores = [dct_score, gwt_score, fep_score, iit_score];
    if (allScores.some(s => typeof s !== 'number' || isNaN(s))) {
        console.error('Invalid theory_fit data in renderStats');
        return;
    }
    const sorted = [...allScores].sort((a, b) => b - a);
    const margin = sorted[0] - sorted[1];
    const winnerName = ['DCT', 'GWT', 'FEP', 'IIT'][allScores.indexOf(sorted[0])];
    const marginOk = margin >= 0.1;
    const silh = typeof data.silhouette_score === 'number' ? data.silhouette_score : null;
    const silhOk = silh !== null && silh > 0.6;

    const rows = [
        ['Winner', winnerName + ' (margin: ' + margin.toFixed(3) + ')', marginOk],
        ['Silhouette', silh !== null ? silh.toFixed(4) : 'N/A', silhOk],
        ['ANOVA p-value', typeof data.anova_p_value === 'number'
            ? data.anova_p_value.toFixed(6) : 'N/A', null],
        ['Margin \u2265 0.1', marginOk ? 'PASS' : 'FAIL', marginOk],
    ];

    const panel = document.getElementById('stats-panel');
    panel.replaceChildren(...rows.map(([label, val, ok]) => {
        const row = document.createElement('div');
        row.className = 'stat-row';
        const lEl = document.createElement('span');
        lEl.className = 'stat-label';
        lEl.textContent = label;
        const vEl = document.createElement('span');
        vEl.className = 'stat-value' +
            (ok === true ? ' pass' : ok === false ? ' fail' : '');
        vEl.textContent = val;
        row.appendChild(lEl);
        row.appendChild(vEl);
        return row;
    }));
}

function renderContrastiveDelta(deltas) {
    const display = document.getElementById('delta-display');
    const label = document.getElementById('delta-label');
    if (!deltas || deltas.length === 0) {
        if (label) label.textContent = 'No contrastive data available';
        return;
    }
    const entry = deltas[0];
    const layers = [entry.early_delta, entry.mid_delta, entry.late_delta];
    const layerNames = ['Early layer', 'Mid layer', 'Late layer'];

    function render(idx) {
        const delta = layers[idx];
        if (!delta) return;
        const l2 = typeof entry.l2_norm === 'number' ? entry.l2_norm.toFixed(4) : 'N/A';
        label.textContent = layerNames[idx] + ' delta — L2: ' + l2;
        const shown = delta.slice(0, 32);
        const maxAbs = Math.max(...shown.map(Math.abs), 1e-10);
        const spans = shown.map((v, i) => {
            const t = (v / maxAbs + 1) / 2;
            const [r, g, b] = coolwarmRgb(t);
            const span = document.createElement('span');
            span.style.background =
                'rgb(' + [r, g, b].map(x => Math.round(x * 255)).join(',') + ')';
            span.style.color = (t > 0.35 && t < 0.65) ? '#333' : '#fff';
            span.textContent = 'N' + i + ': ' + v.toFixed(3);
            return span;
        });
        display.replaceChildren(...spans);
    }

    render(parseInt(document.getElementById('delta-slider').value, 10));
    document.getElementById('delta-slider').addEventListener('input', e => {
        render(parseInt(e.target.value, 10));
    });
}

function initExplorer() {
    const btn = document.getElementById('explorer-btn');
    const input = document.getElementById('explorer-input');
    if (!btn || !input) return;

    btn.addEventListener('click', () => runExplorer(input.value.trim()));
    input.addEventListener('keydown', e => {
        if (e.key === 'Enter') runExplorer(input.value.trim());
    });
}

function runExplorer(text) {
    if (!text) return;
    const label = document.getElementById('explorer-label');
    const display = document.getElementById('explorer-display');
    if (!label || !display) return;
    label.textContent = 'Running...';
    display.replaceChildren();

    fetch('/api/explore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    })
    .then(r => r.json())
    .then(result => {
        if (!result.mid || !Array.isArray(result.mid)) {
            label.textContent = 'Invalid response from server';
            return;
        }
        const layerData = result.mid;
        const maxVal = Math.max(...layerData.map(Math.abs), 1e-10);
        label.textContent = 'Mid-layer activation for: "' + result.input_text + '"';
        const spans = layerData.slice(0, 32).map((v, i) => {
            const t = (v / maxVal + 1) / 2;
            const [r, g, b] = coolwarmRgb(t);
            const span = document.createElement('span');
            span.style.background =
                'rgb(' + [r, g, b].map(x => Math.round(x * 255)).join(',') + ')';
            span.style.color = (t > 0.35 && t < 0.65) ? '#333' : '#fff';
            span.textContent = 'N' + i + ': ' + v.toFixed(3);
            return span;
        });
        display.replaceChildren(...spans);
    })
    .catch(err => {
        console.error('Explorer error:', err);
        if (label) label.textContent = 'Error: ' + err.message;
    });
}
