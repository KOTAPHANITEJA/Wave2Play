function updateStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('status').innerHTML = `
                Playing: ${data.playing}<br>
                Volume: ${data.volume}%<br>
                Current Track: ${data.track}<br>
                Mode: ${data.mode}<br>
                Emergency: ${data.emergency ? 'Active' : 'Inactive'}
            `;
        });
}

function control(action) {
    fetch(`/api/control/${action}`, {
        method: 'POST'
    }).then(() => updateStatus());
}

// Update status every second
setInterval(updateStatus, 1000);
updateStatus();