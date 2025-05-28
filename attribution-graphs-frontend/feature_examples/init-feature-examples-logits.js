window.initFeatureExamplesExtra = function({renderAll, visState, sel}) {
  renderAll.feature.fns.push(async () => {
    let { top_logits, bottom_logits } = visState.feature;
    if (!top_logits?.length && !bottom_logits?.length) return;

    var containerSel = sel.select('.feature-example-logits').html('')
    containerSel.append('div.section-title').text('Token Predictions')
    // containerSel.parent().st({position: 'sticky', top: 0})

    for (let [rowName, logits] of [['Top', top_logits], ['Bottom', bottom_logits]]) {
      if (!logits?.length) continue;

      var row = containerSel.append('div.ctx-container')
      row.append('div.logit-label').text(rowName)

      row.appendMany('span.token', logits).text(d => d)
    }

    let { self_explanation_enc, self_explanation_dec } = visState.feature;
    if (!self_explanation_enc?.length && !self_explanation_dec?.length) return;

    var containerSel = sel.select('.feature-example-self-explanation').html('')

    for (let [rowName, explanations] of [['Encoder', self_explanation_enc], ['Decoder', self_explanation_dec]]) {
      if (!explanations?.length) continue;

      var row = containerSel.append('div.ctx-container')
      row.append('div.section-title').text(`Self-Explanation (${rowName})`)
      row.appendMany('div.logit-label', explanations).text(d => d)
    }
  })
}

window.init?.()
