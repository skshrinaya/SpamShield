document.getElementById('checkSms').addEventListener('click', () => {
    const smsInput = document.getElementById('smsInput').value;
    if (smsInput) {
        fetch('http://localhost:5000/predict_sms', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ input_sms: smsInput })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('smsResult').innerText = data.result;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('smsResult').innerText = 'Error occurred';
        });
    }
});

document.getElementById('checkUrl').addEventListener('click', () => {
    const urlInput = document.getElementById('urlInput').value;
    if (urlInput) {
        fetch('http://localhost:5000/predict_url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ input_url: urlInput })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('urlResult').innerText = data.result;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('urlResult').innerText = 'Error occurred';
        });
    }
});
