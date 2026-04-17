// viz/brain_heatmap/js/overlay.js
export function initOverlay(vizData) {
    const checkbox = document.getElementById('overlay-toggle');
    checkbox.addEventListener('change', () => renderPanel(vizData));
    renderPanel(vizData);
}

function renderPanel(vizData) {
    const { dct_score, gwt_score, fep_score, iit_score } = vizData.theory_fit;
    const scores = { DCT: dct_score, GWT: gwt_score, FEP: fep_score, IIT: iit_score };
    const winner = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];

    const panel = document.getElementById('theory-panel');
    const rows = Object.entries(scores).map(([name, score]) => {
        const row = document.createElement('div');
        row.className = 'theory-row';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'theory-name' + (name === winner ? ' theory-winner' : '');
        nameSpan.textContent = name === winner ? name + ' \u2605' : name;

        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'theory-score' + (name === winner ? ' theory-winner' : '');
        scoreSpan.textContent = score.toFixed(3);

        row.appendChild(nameSpan);
        row.appendChild(scoreSpan);
        return row;
    });

    panel.replaceChildren(...rows);
}
