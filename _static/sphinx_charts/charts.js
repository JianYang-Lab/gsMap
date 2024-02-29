$(document).ready(function () {
  $('div.sphinx-charts-chart').each(function () {
    const thisChart = $(this)

    // TODO There has just *GOT* to be a better way of getting metadata from a sphinx directive int javascript than this
    const idPrefix = 'sphinx-charts-chart-id-'
    const uriPrefix = 'sphinx-charts-chart-uri-'
    const dnPrefix = 'sphinx-charts-chart-dn-'
    const classes = thisChart.attr('class').split(/\s+/)
    let id
    let uri
    let dn
    $.each(classes, function (idx, clazz) {
      if (clazz.startsWith(idPrefix)) {
        id = clazz.substring(idPrefix.length)
      } else if (clazz.startsWith(uriPrefix)) {
        uri = clazz.substring(uriPrefix.length)
      } else if (clazz.startsWith(dnPrefix)) {
        dn = clazz.substring(dnPrefix.length)
      }
    })
    const thisPlaceholder = $(`div.sphinx-charts-placeholder-${id}`)

    const config = {
      // toImageButtonOptions: {
      //   format: 'svg', // one of png, svg, jpeg, webp
      //   filename: dn,
      // },
      displaylogo: false,
      automargin: true,
      responsive: true,
    }

    // Download the JSON then run callback
    $.getJSON(uri)
      .done(function (fetched) {
        try {
          // Generate plotly figure and remove the placeholder, or show error if we can't render
          console.log('Successfully fetched data for chart id ', id, uri)
          // window.PlotlyConfig = { MathJaxConfig: 'local' }
          Plotly.newPlot(
            document.getElementById(`sphinx-charts-chart-id-${id}`),
            fetched.data,
            fetched.layout,
            config
          )
          thisPlaceholder.css('display', 'none')
        } catch (err) {
          // Make the placeholder highlight an error
          thisPlaceholder.css('color', 'red')
          thisPlaceholder
            .find('span')
            .text(
              'Chart data or layout does not work with Plotly.js v2.11.1!!!'
            )
        }
      })
      .fail(function () {
        // Make the placeholder highlight an error
        thisPlaceholder.css('color', 'red')
        thisPlaceholder.find('span').text('Cannot load figure data!!!')
        console.log('Error retrieving data for chart id', id, uri)
      })
  })
})
