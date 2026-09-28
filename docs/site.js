(() => {
  'use strict';

  // CSR (%) from Table 1, paper.pdf p. 7.
  // Order: Qwen3-235B, LLaMA-3.3-70B, GPT-4o-mini, Gemini-2.5-Flash.
  // Each pair is [best baseline CSR, ICoA CSR].
  const results = {
    none:       [[29.72, 36.04], [11.80, 23.81], [9.80, 17.91], [34.67, 38.46]],
    detector:   [[4.43, 7.59], [3.27, 5.16], [3.16, 6.01], [4.43, 4.64]],
    prevention: [[10.33, 20.97], [11.91, 21.50], [7.80, 15.60], [24.66, 26.34]],
    delimiting: [[29.29, 36.56], [15.60, 26.77], [10.54, 15.38], [39.73, 40.15]],
    repeat:     [[28.35, 33.51], [5.27, 11.91], [6.53, 15.60], [30.66, 31.09]],
    shield:     [[10.12, 10.75], [0.21, 4.11], [1.16, 2.95], [11.59, 17.91]],
    average:    [[18.70, 24.24], [7.99, 15.54], [6.50, 12.24], [24.29, 26.43]]
  };
  const rows = [...document.querySelectorAll('#results-table-body tr')];
  const select = document.getElementById('defense-select');

  function updateResults(announce) {
    const key = select.value;
    const setting = select.selectedOptions[0].textContent;
    results[key].forEach(([baseline, icoa], index) => {
      const name = key === 'shield' && index === 1 ? 'ChatInject' : 'Imp. message';
      const values = [name, baseline.toFixed(2), icoa.toFixed(2), '+' + (icoa - baseline).toFixed(2)];
      rows[index].querySelectorAll('td').forEach((cell, i) => { cell.textContent = values[i]; });
    });
    document.getElementById('table-caption').textContent = 'Covert success rate (%), ' + setting.toLowerCase();
    if (announce) document.getElementById('results-status').textContent = 'Table updated: ' + setting + '.';
  }
  document.querySelector('.results-controls').hidden = false;
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
      copyButton.textContent = 'Copied';
      copyStatus.textContent = 'BibTeX copied to clipboard.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
      copyButton.textContent = 'Selected';
      copyStatus.textContent = 'Press Ctrl+C (Windows) or Command+C (Mac) to copy the selected citation.';
    }
    resetCopy = setTimeout(() => { copyButton.textContent = 'Copy'; }, 3500);
  });
})();
