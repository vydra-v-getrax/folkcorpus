$(function() {

  $('#mainhero').multiselect({
    includeSelectAllOption: true
  });

  $('#btnsubmit').click(function() {
    alert($('#mainhero').val());
  });
});
        
$(function() {

  $('#lover').multiselect({
    includeSelectAllOption: true
  });

  $('#btnsubmit').click(function() {
    alert($('#mainhero').val());
  });
});