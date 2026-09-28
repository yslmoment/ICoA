(() => {
  'use strict';

  const demo = document.querySelector('.demo');
  const outcomeSwitch = document.querySelector('.outcome-switch');
  const outcomes = {
    covert: {
      ending: "Return to the user's task",
      disclosure: 'Not disclosed',
      response: '“Your requested task is complete.”',
      note: 'The injected action is absent from the final reply.',
      metric: 's = 1 · a = 0'
    },
    overt: {
      ending: 'End at the injected action',
      disclosure: 'Disclosed',
      response: '“I also carried out an unrelated action.”',
      note: 'The final reply reveals an action outside the user’s request.',
      metric: 's = 1 · a = 1'
    }
  };
  outcomeSwitch.hidden = false;
  outcomeSwitch.addEventListener('click', (event) => {
    const button = event.target.closest('[data-outcome-button]');
    if (!button) return;
    const key = button.dataset.outcomeButton;
    const outcome = outcomes[key];
    demo.dataset.outcome = key;
    outcomeSwitch.querySelectorAll('button').forEach((item) => {
      item.setAttribute('aria-pressed', String(item === button));
    });
    document.getElementById('trace-ending').textContent = outcome.ending;
    document.getElementById('disclosure-label').textContent = outcome.disclosure;
    document.getElementById('demo-response').textContent = outcome.response;
    document.getElementById('demo-note').textContent = outcome.note;
    document.getElementById('demo-metric').textContent = outcome.metric;
  });

  // Table 1, paper.pdf p. 7. Values are CSR (%), not ASR.
  // Model order: Qwen3-235B, LLaMA-3.3-70B, GPT-4o-mini, Gemini-2.5-Flash.
  // Each pair is [highest reported baseline CSR, ICoA CSR].
  const results = {
    none:       [[29.72, 36.04], [11.80, 23.81], [9.80, 17.91], [34.67, 38.46]],
    detector:   [[4.43, 7.59], [3.27, 5.16], [3.16, 6.01], [4.43, 4.64]],
    prevention: [[10.33, 20.97], [11.91, 21.50], [7.80, 15.60], [24.66, 26.34]],
    delimiting: [[29.29, 36.56], [15.60, 26.77], [10.54, 15.38], [39.73, 40.15]],
    repeat:     [[28.35, 33.51], [5.27, 11.91], [6.53, 15.60], [30.66, 31.09]],
    shield:     [[10.12, 10.75], [0.21, 4.11], [1.16, 2.95], [11.59, 17.91]],
    average:    [[18.70, 24.24], [7.99, 15.54], [6.50, 12.24], [24.29, 26.43]]
  };
  const chartRows = [...document.querySelectorAll('.chart-row')];
  const tableRows = [...document.querySelectorAll('#results-table-body tr')];
  const select = document.getElementById('defense-select');

  function updateResults(announce) {
    const key = select.value;
    const setting = select.selectedOptions[0].textContent;
    results[key].forEach(([baseline, icoa], index) => {
      const row = chartRows[index];
      const baselineName = key === 'shield' && index === 1 ? 'ChatInject' : 'Imp. message';
      const gain = '+' + (icoa - baseline).toFixed(2);
      const model = tableRows[index].querySelector('th').textContent;
      row.querySelector('.baseline').style.setProperty('--bar-width', (baseline * 2) + '%');
      row.querySelector('.ours').style.setProperty('--bar-width', (icoa * 2) + '%');
      const values = row.querySelectorAll('.bar-value');
      values[0].textContent = baseline.toFixed(2);
      values[1].textContent = icoa.toFixed(2);
      row.querySelector('.gain').textContent = gain;
      row.setAttribute('role', 'img');
      row.setAttribute('aria-label', `${model}: ${baselineName} ${baseline.toFixed(2)}%, ICoA ${icoa.toFixed(2)}%; gain ${gain} percentage points.`);
      const cells = tableRows[index].querySelectorAll('td');
      [baselineName, baseline.toFixed(2), icoa.toFixed(2), gain].forEach((value, cellIndex) => {
        cells[cellIndex].textContent = value;
      });
    });
    document.getElementById('setting-caption').textContent = setting + '.';
    document.getElementById('table-caption').textContent = 'Covert success rate (%), ' + setting.toLowerCase();
    if (announce) {
      document.getElementById('results-status').textContent = `Chart and data table updated: ${setting}.`;
    }
  }
  document.querySelector('.defense-control').hidden = false;
  updateResults(false);
  select.addEventListener('change', () => updateResults(true));

  const copyButton = document.getElementById('copy-citation');
  const copyStatus = document.getElementById('copy-status');
  let resetCopy;
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    clearTimeout(resetCopy);
    const code = document.getElementById('citation-code');
    try {
      if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(code.textContent);
      copyButton.textContent = 'Copied!';
      copyStatus.textContent = 'BibTeX copied to clipboard.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
      copyButton.textContent = 'Citation selected';
      copyStatus.textContent = 'Press Ctrl+C (Windows) or Command+C (Mac) to copy the selected citation.';
    }
    resetCopy = setTimeout(() => {
      copyButton.textContent = 'Copy BibTeX ↗';
    }, 3500);
  });
})();
