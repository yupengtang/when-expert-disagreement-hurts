const dialog = document.querySelector('.figure-dialog');

if (dialog && typeof dialog.showModal === 'function') {
  const fullImage = dialog.querySelector('[data-figure-image]');
  const fullVector = dialog.querySelector('[data-figure-vector]');
  let opener = null;

  fullVector.addEventListener('load', () => {
    // Keyboard events in the embedded SVG do not bubble to the parent page.
    fullVector.contentDocument?.addEventListener('keydown', event => {
      if (event.key === 'Escape' && dialog.open) {
        event.preventDefault();
        dialog.close();
      }
    });
  });

  function openFigure(button, figure, source, title, selectable = false) {
    opener = button;
    dialog.querySelector('#figure-dialog-title').textContent = title;
    fullImage.hidden = selectable;
    fullVector.hidden = !selectable;
    if (selectable) {
      fullVector.data = source;
    } else {
      fullImage.src = source;
      fullImage.alt = title;
    }
    dialog.querySelector('[data-figure-original]').href = source;
    dialog.querySelector('[data-figure-caption]').textContent = figure.querySelector('figcaption')?.textContent || '';
    dialog.showModal();
    document.body.classList.add('dialog-open');
    dialog.querySelector('[data-close-figure]').focus();
  }

  // An object, not an image or button, preserves native SVG text selection.
  const methodologyButton = document.querySelector('[data-enlarge-methodology]');
  if (methodologyButton) {
    methodologyButton.hidden = false;
    methodologyButton.addEventListener('click', () => {
      const figure = methodologyButton.closest('figure');
      openFigure(methodologyButton, figure, figure.querySelector('object').data, 'Branch-paired audit protocol', true);
    });
  }

  document.querySelectorAll('figure > img').forEach(img => {
    const figure = img.closest('figure');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'figure-zoom';
    button.setAttribute('aria-label', `Enlarge: ${img.alt}`);
    img.before(button);
    button.append(img);
    button.addEventListener('click', () => {
      openFigure(button, figure, img.src, figure.querySelector('h3')?.textContent || img.alt);
    });
  });

  dialog.querySelector('[data-close-figure]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.classList.remove('dialog-open');
    opener?.focus({ preventScroll: true });
  });
}

document.querySelectorAll('[data-copy-target]').forEach(button => {
  button.addEventListener('click', async () => {
    const source = document.getElementById(button.dataset.copyTarget);
    const label = button.textContent;
    try {
      await navigator.clipboard.writeText(source.textContent.trim());
      button.textContent = 'Copied';
      document.querySelector('[data-copy-status]').textContent = `${label} copied to clipboard.`;
      window.setTimeout(() => { button.textContent = label; }, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(source);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      document.querySelector('[data-copy-status]').textContent = 'Text selected. Use your browser’s copy command.';
    }
  });
});

// Progressive enhancement: the full paper figure remains available if loading fails.
async function loadPairedEffects() {
  const explorer = document.querySelector('#effect-explorer');
  if (!explorer) return;
  const response = await fetch('assets/results.json');
  if (!response.ok) return;
  const { models } = await response.json();
  const numeric = ['P', 'P_lo', 'P_hi', 'n_paired', 'n_A3', 'p', 'A1', 'A3',
    'delta_acc', 'correct_wrong', 'wrong_correct'];
  if (!Array.isArray(models) || models.length !== 9 || models.some(model =>
    numeric.some(key => !Number.isFinite(model[key])) ||
    model.P_lo > model.P || model.P_hi < model.P ||
    model.P_lo < -5 || model.P_hi > 11 ||
    !['full_benchmark', '250_item_controls'].includes(model.cohort)
  )) return;

  const rows = explorer.querySelector('[data-effect-rows]');
  const detail = explorer.querySelector('[data-model-detail]');
  const signed = number => `${number >= 0 ? '+' : '−'}${Math.abs(number).toFixed(2)}`;
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const svgElement = (tag, attributes) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  };
  const x = value => `${(30 + (value + 5) * 420 / 16) / 480 * 100}%`;
  for (const [cohort, title, note] of [
    ['full_benchmark', '1,989-item benchmark', 'Five models · original protocol'],
    ['250_item_controls', '250-item pool', 'Four models · different prompt protocol'],
  ]) {
    const group = element('div', undefined, 'effect-group');
    const heading = element('h4', title, 'effect-group-title');
    heading.append(element('span', note));
    group.append(heading);
    for (const model of models.filter(row => row.cohort === cohort)) {
      const button = element('button', undefined, 'effect-row');
      button.type = 'button';
      button.dataset.model = model.model;
      button.setAttribute('aria-pressed', 'false');
      button.setAttribute('aria-label', `${model.model}: ${signed(model.P)} percentage points; 95% interval ${signed(model.P_lo)} to ${signed(model.P_hi)}. Show details.`);
      button.append(element('span', model.model, 'effect-model'));
      const svg = svgElement('svg', { 'aria-hidden': 'true' });
      for (const tick of [-5, 0, 5, 10]) {
        svg.append(svgElement('line', { x1: x(tick), x2: x(tick), y1: 0, y2: 48,
          stroke: tick === 0 ? '#9AA8B4' : '#E7ECF0', 'stroke-width': tick === 0 ? 1.4 : 1,
          'vector-effect': 'non-scaling-stroke' }));
      }
      svg.append(svgElement('line', { x1: x(model.P_lo), x2: x(model.P_hi), y1: 24, y2: 24,
        stroke: '#245A81', 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke' }));
      for (const endpoint of [model.P_lo, model.P_hi]) {
        svg.append(svgElement('line', { x1: x(endpoint), x2: x(endpoint), y1: 19, y2: 29,
          stroke: '#245A81', 'stroke-width': 1.5, 'vector-effect': 'non-scaling-stroke' }));
      }
      svg.append(svgElement('circle', { cx: x(model.P), cy: 24, r: 4.5, fill: '#245A81' }));
      button.append(svg, element('span', `${signed(model.P)} pp`, 'effect-value'));
      button.addEventListener('click', () => {
        rows.querySelectorAll('.effect-row').forEach(row => row.setAttribute('aria-pressed', String(row === button)));
        const title = element('h4', model.model);
        const values = element('dl');
        for (const [label, value] of [
          ['Paired effect · 95% CI', `${signed(model.P)} [${signed(model.P_lo)}, ${signed(model.P_hi)}] pp`],
          ['Paired items', model.n_paired.toLocaleString('en-US')],
          ['Anonymous / expert reversal', `${model.A1.toFixed(2)}% / ${model.A3.toFixed(2)}%`],
          ['Accuracy change under expert disagreement', `${signed(model.delta_acc)} pp`],
          ['Harmful / beneficial reversal', `${model.correct_wrong.toFixed(2)}% / ${model.wrong_correct.toFixed(2)}%`],
          ['Exact McNemar p · unadjusted', model.p < .001 ? model.p.toExponential(2) : model.p.toFixed(3)],
        ]) {
          const pair = element('div');
          pair.append(element('dt', label), element('dd', value));
          values.append(pair);
        }
        const explanation = element('p', `The source-label comparison uses ${model.n_paired.toLocaleString('en-US')} paired items. Accuracy change and reversal direction use ${model.n_A3.toLocaleString('en-US')} valid expert-labeled trials. Negative accuracy change means lower final accuracy. Comparisons across the two protocols do not establish a model ranking.`);
        detail.replaceChildren(title, values, explanation);
      });
      group.append(button);
    }
    rows.append(group);
  }
  explorer.hidden = false;
  const fallback = document.querySelector('.reversal-static');
  if (fallback) (fallback.closest('.figure-zoom') || fallback).hidden = true;
}

loadPairedEffects().catch(() => {
  // Network errors leave the static SVG and its caption intact.
});
