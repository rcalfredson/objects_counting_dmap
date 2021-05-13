setTimeout(() => {
    console.log('This message should appear.');
}, 1500);

var badFunc = function () {
    throw 'Something bad happened.';
}
badFunc();