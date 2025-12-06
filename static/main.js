document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('audio-upload');
    const analyzeBtn = document.getElementById('analyze-btn');
    const fileNameDisplay = document.getElementById('file-name');
    const resultsSection = document.getElementById('results-section');
    const demoBtn = document.getElementById('demo-btn');

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

    demoBtn.addEventListener('click', () => {
        // Mock demo data for visual confirmation
        simulateDemoAnalysis();
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

        // Update Diagnosis
        const diagTitle = document.getElementById('diagnosis-title');
        const diagDesc = document.getElementById('diagnosis-desc');
        const statusIcon = document.getElementById('status-icon');

        diagTitle.textContent = data.diagnosis;
        diagDesc.textContent = data.description;

        // Reset classes
        statusIcon.className = 'status-icon';
        statusIcon.classList.add(`status-${data.status}`);

        // Render Waveform
        renderChart(data.waveform);
    }

    function renderChart(waveform) {
        const ctx = document.getElementById('waveformChart').getContext('2d');

        if (chartInstance) {
            chartInstance.destroy();
            chartInstance = null; // Explicitly nullify
        }

        // Reset context state safely
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        // Create gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(0, 242, 255, 0.5)');
        gradient.addColorStop(1, 'rgba(0, 242, 255, 0)');

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({ length: waveform.length }, (_, i) => i),
                datasets: [{
                    label: 'Amplitude',
                    data: waveform,
                    borderColor: '#00f2ff',
                    backgroundColor: gradient,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.4
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

    function simulateDemoAnalysis() {
        setLoading(true);
        resultsSection.classList.add('hidden');

        setTimeout(() => {
            const mockData = {
                bpm: 72,
                hrv_sdnn: 45.2,
                diagnosis: "Normal Sinus Rhythm",
                description: "This is a demo result. Regular heartbeat pattern detected with optimal variability.",
                status: "healthy",
                waveform: Array.from({ length: 100 }, () => Math.random() * 0.5 - 0.25)
            };
            displayResults(mockData);
            setLoading(false);
            fileNameDisplay.textContent = "Demo Mode Active";
        }, 1500);
    }
});
