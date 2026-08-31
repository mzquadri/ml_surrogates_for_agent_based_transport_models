const results = [
  [10, 316350, 1.0555], [25, 790875, 1.7926], [30, 949050, 1.9218],
  [40, 1265400, 2.1333], [50, 1581750, 2.3197], [60, 1898100, 2.5008],
  [70, 2214450, 2.6893], [75, 2372625, 2.7945], [80, 2530800, 2.9131],
  [85, 2688975, 3.0514], [90, 2847150, 3.2257], [95, 3005325, 3.4787],
  [100, 3163500, 3.9476],
];

const retention = document.querySelector('#retention');
const number = new Intl.NumberFormat('en-US');
const byRetention = (index) => results[index];

function drawChart(selected, selectedPoint) {
  const svg = document.querySelector('#risk-chart');
  const width = 700, height = 310, left = 56, right = 22, top = 22, bottom = 42;
  const x = (value) => left + ((value - 10) / 90) * (width - left - right);
  const y = (value) => top + ((4.1 - value) / 3.2) * (height - top - bottom);
  const points = results.map(([percent,, mae]) => `${x(percent)},${y(mae)}`).join(' ');
  const grid = [1, 2, 3, 4].map((value) => `<line x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}" /><text x="${left - 12}" y="${y(value) + 4}" text-anchor="end">${value}</text>`).join('');
  const labels = [10, 50, 100].map((value) => `<text x="${x(value)}" y="${height - 14}" text-anchor="middle">${value}%</text>`).join('');
  svg.innerHTML = `<g class="grid">${grid}</g><text class="axis-label" x="${left}" y="13">MAE (veh/h)</text><polyline class="line" points="${points}" /><line class="selected-line" x1="${x(selected)}" y1="${top}" x2="${x(selected)}" y2="${height - bottom}" /><circle class="selected-dot" cx="${x(selectedPoint[0])}" cy="${y(selectedPoint[2])}" r="7" /><text class="callout" x="${Math.min(x(selected) + 12, 600)}" y="${y(selectedPoint[2]) - 13}">${selectedPoint[2].toFixed(2)} MAE</text><text class="axis-label" x="${width - right}" y="${height - 14}" text-anchor="end">retained predictions</text>${labels}`;
}

function update() {
  const selectedIndex = Number(retention.value);
  const [selected, accepted, mae] = byRetention(selectedIndex);
  const total = results.at(-1)[1];
  const reviewed = total - accepted;
  const reduction = ((1 - mae / results.at(-1)[2]) * 100);
  document.querySelector('#retention-value').value = `${selected}%`;
  document.querySelector('#mae').textContent = mae.toFixed(2);
  document.querySelector('#reduction').textContent = selected === 100 ? 'Baseline: all predictions are accepted' : `${reduction.toFixed(1)}% lower than accepting all predictions`;
  document.querySelector('#accepted').textContent = number.format(accepted);
  document.querySelector('#reviewed').textContent = number.format(reviewed);
  document.querySelector('#accepted-share').textContent = `${selected}% of the held-out predictions`;
  document.querySelector('#reviewed-share').textContent = `${100 - selected}% of the held-out predictions`;
  drawChart(selected, [selected, accepted, mae]);
}

retention.addEventListener('input', update);
update();
