document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('audio-upload');
    const analyzeBtn = document.getElementById('analyze-btn');
    const fileNameDisplay = document.getElementById('file-name');
    const resultsSection = document.getElementById('results-section');

    let chartInstance = null;
    let selectedFiles = null;

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFiles = e.target.files;

            if (selectedFiles.length > 1) {
                fileNameDisplay.textContent = `Selected: ${selectedFiles.length} files`;
            } else {
                fileNameDisplay.textContent = `Selected: ${selectedFiles[0].name}`;
            }
            analyzeBtn.disabled = false;
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFiles || selectedFiles.length === 0) return;

        setLoading(true);
        resultsSection.classList.add('hidden');

        // Setup FormData with all files
        const formData = new FormData();
        let audioFileForPlayer = null;

        for (let i = 0; i < selectedFiles.length; i++) {
            formData.append('audio', selectedFiles[i]);

            // Try to find a playable audio file for the player
            if (selectedFiles[i].type.startsWith('audio/') || selectedFiles[i].name.endsWith('.wav') || selectedFiles[i].name.endsWith('.mp3')) {
                audioFileForPlayer = selectedFiles[i];
            }
        }

        // Set up audio player if possible
        const audioPlayer = document.getElementById('audio-player');
        if (audioFileForPlayer) {
            const fileURL = URL.createObjectURL(audioFileForPlayer);
            audioPlayer.src = fileURL;
            audioPlayer.style.display = 'block';
        } else {
            audioPlayer.style.display = 'none';
        }

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Analysis failed');
            }

            displayResults(data);
        } catch (error) {
            console.error(error);
            alert(`Analysis Error: ${error.message}`);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        const btnText = analyzeBtn.querySelector('.btn-text');
        const loader = analyzeBtn.querySelector('.loader-ring');

        if (isLoading) {
            btnText.classList.add('hidden');
            loader.classList.remove('hidden');
            analyzeBtn.disabled = true;
        } else {
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    }

    function displayResults(data) {
        resultsSection.classList.remove('hidden');

        // Count up animation for numbers
        animateValue("bpm-value", 0, data.bpm, 1000);
        animateValue("hrv-value", 0, data.hrv_sdnn, 1000);

        // New Metrics
        const rmssdVal = document.getElementById('rmssd-value');
        const snrVal = document.getElementById('snr-value');
        if (rmssdVal) rmssdVal.textContent = data.hrv_rmssd;
        if (snrVal) snrVal.textContent = data.snr + ' dB';

        // Update Diagnosis
        const diagTitle = document.getElementById('diagnosis-title');
        const diagDesc = document.getElementById('diagnosis-desc');
        const statusIcon = document.getElementById('status-icon');

        diagTitle.textContent = data.diagnosis;
        diagDesc.textContent = data.description;

        // Reset classes
        statusIcon.className = 'status-dot';
        if (data.status === 'danger') statusIcon.classList.add('status-danger');
        else if (data.status === 'warning') statusIcon.classList.add('status-warning');
        else statusIcon.classList.add('status-healthy');

        // Render Waveform
        renderChart(data.waveform);
    }

    function renderChart(waveform) {
        const ctx = document.getElementById('waveformChart').getContext('2d');

        if (chartInstance) {
            chartInstance.destroy();
            chartInstance = null;
        }

        // Reset context state safely
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({ length: waveform.length }, (_, i) => i),
                datasets: [{
                    label: 'Signal',
                    data: waveform,
                    borderColor: '#ffffff',
                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false, /* Minimal line only */
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { display: false },
                    y: {
                        display: false,
                        min: -1,
                        max: 1
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    function animateValue(id, start, end, duration) {
        const obj = document.getElementById(id);
        if (!obj) return;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end;
            }
        };
        window.requestAnimationFrame(step);
    }
});
