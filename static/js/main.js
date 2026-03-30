document.addEventListener('DOMContentLoaded', () => {
    // Generate animated wind lines
    const container = document.getElementById('windLines');
    if (container) {
        for (let i = 0; i < 15; i++) {
            const line = document.createElement('div');
            line.className = 'wind-line';
            line.style.top = `${Math.random() * 100}%`;
            line.style.left = `${Math.random() * 100}%`;
            line.style.width = `${Math.random() * 100 + 40}px`;
            line.style.animationDuration = `${Math.random() * 3 + 2}s`;
            line.style.animationDelay = `${Math.random() * 5}s`;
            container.appendChild(line);
        }
    }

    // Initialize DateTime with current local time
    const timestampInput = document.getElementById('timestamp');
    if (timestampInput) {
        const now = new Date();
        const tzoffset = now.getTimezoneOffset() * 60000;
        const localISOTime = (new Date(now - tzoffset)).toISOString().slice(0, 16);
        timestampInput.value = localISOTime;
    }

    // ── Validation Logic ──
    const FIELDS = [
        { id: 'wind_speed_ms', min: 0, max: 100, name: 'Wind Speed' },
        { id: 'theoretical_power', min: 0, max: 10000, name: 'Theoretical Power' },
        { id: 'wind_direction', min: 0, max: 360, name: 'Wind Direction' }
    ];

    function validateField(field) {
        const input = document.getElementById(field.id);
        if (!input) return null;
        const val = parseFloat(input.value);
        if (isNaN(val)) return `Please enter ${field.name}`;
        if (val < field.min) return `${field.name} cannot be less than ${field.min}`;
        if (val > field.max) return `${field.name} cannot exceed ${field.max}`;
        return null;
    }

    function setFieldState(field, error) {
        const input = document.getElementById(field.id);
        if (!input) return;
        const wrap = input.closest('.form-group');
        let errorEl = wrap.querySelector('.field-error');

        if (error) {
            input.classList.remove('is-valid');
            input.classList.add('is-invalid');
            if (!errorEl) {
                errorEl = document.createElement('div');
                errorEl.className = 'field-error';
                wrap.appendChild(errorEl);
            }
            errorEl.textContent = error;
        } else {
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
            if (errorEl) errorEl.remove();
        }
    }

    FIELDS.forEach(field => {
        const input = document.getElementById(field.id);
        if (!input) return;

        input.addEventListener('keypress', (e) => {
            const allowed = ['Backspace', 'Delete', 'ArrowLeft', 'ArrowRight', 'Tab', '.', 'Enter'];
            if (e.key === '-' && input.selectionStart === 0) return;
            if (e.ctrlKey || e.metaKey) return;
            if (/^\d$/.test(e.key)) return;
            if (allowed.includes(e.key)) return;
            e.preventDefault();
        });

        input.addEventListener('blur', () => {
            const error = validateField(field);
            setFieldState(field, error);
        });
        input.addEventListener('input', () => {
            if (input.classList.contains('is-invalid')) {
                const error = validateField(field);
                setFieldState(field, error);
            }
        });
    });

    const summaryEl = document.createElement('div');
    summaryEl.className = 'form-error-summary';
    summaryEl.innerHTML = '⚠ &nbsp;<span id="summaryText"></span>';
    const btnWrap = document.querySelector('.btn-wrap');
    if (btnWrap) btnWrap.before(summaryEl);

    // ── Form submission ──
    const predictForm = document.getElementById('predictForm');
    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            let firstError = null;
            let errorCount = 0;
            FIELDS.forEach(field => {
                const error = validateField(field);
                setFieldState(field, error);
                if (error) {
                    errorCount++;
                    if (!firstError) firstError = field.id;
                }
            });

            if (errorCount > 0) {
                document.getElementById('summaryText').textContent =
                    `Please fix ${errorCount} field${errorCount > 1 ? 's' : ''} before predicting.`;
                summaryEl.classList.add('visible');
                document.getElementById(firstError).scrollIntoView({ behavior: 'smooth', block: 'center' });
                const card = document.querySelector('.card');
                if (card) {
                    card.classList.remove('shake');
                    void card.offsetWidth;
                    card.classList.add('shake');
                }
                return;
            }

            summaryEl.classList.remove('visible');
            const btn = document.getElementById('submitBtn');
            const resultDiv = document.getElementById('result');
            btn.classList.add('loading');
            btn.disabled = true;
            resultDiv.style.display = 'none';

            try {
                const response = await fetch('/predict_smart', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        wind_speed_ms: parseFloat(document.getElementById('wind_speed_ms').value),
                        theoretical_power_kwh: parseFloat(document.getElementById('theoretical_power').value),
                        wind_direction: parseFloat(document.getElementById('wind_direction').value),
                        timestamp: document.getElementById('timestamp').value
                    })
                });

                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    let errMsg = `Server error: ${response.status}`;
                    if (errData.detail) {
                        if (Array.isArray(errData.detail)) {
                            errMsg = errData.detail.map(e => `${e.loc.join('.')}: ${e.msg}`).join(' | ');
                        } else {
                            errMsg = errData.detail;
                        }
                    }
                    throw new Error(errMsg);
                }
                const data = await response.json();
                renderResults(data.predictions);
            } catch (error) {
                console.error("Smart Prediction error:", error);
                resultDiv.innerHTML = `<div class="error-box">⚠ ${error.message}</div>`;
                resultDiv.style.display = 'block';
            } finally {
                btn.classList.remove('loading');
                btn.disabled = false;
            }
        });
    }

    let chartInstance = null;

    function renderResults(predictions) {
        const resultDiv = document.getElementById('result');
        const values = Object.values(predictions);
        const ensembleValue = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1);
        let cardsHtml = '';
        let labels = [];
        let chartData = [];
        let delay = 0.05;

        for (const [model, value] of Object.entries(predictions)) {
            labels.push(model);
            chartData.push(Number(value).toFixed(1));
            cardsHtml += `
            <div class="model-card" style="animation-delay:${delay}s">
                <div class="model-name">${model}</div>
                <div class="model-value">${Number(value).toFixed(1)}</div>
                <div class="model-unit">KW</div>
            </div>`;
            delay += 0.07;
        }

        resultDiv.innerHTML = `
            <div class="result-header">
                <div class="result-header-dot"></div>
                <span class="result-header-text">Prediction Intelligence</span>
            </div>
            <div class="ensemble-card">
                <div class="ensemble-label">Ensemble Forecast</div>
                <div class="ensemble-value">${ensembleValue}</div>
                <div class="ensemble-unit">KILOWATTS</div>
            </div>
            <div class="result-grid">
                ${cardsHtml}
            </div>
            <div class="chart-container">
                <canvas id="predictionChart"></canvas>
            </div>
        `;
        resultDiv.style.display = 'block';

        const ctx = document.getElementById('predictionChart').getContext('2d');
        if (chartInstance) chartInstance.destroy();
        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Predicted Power (KW)',
                    data: chartData,
                    backgroundColor: 'rgba(6, 182, 212, 0.4)',
                    borderColor: 'rgba(6, 182, 212, 1)',
                    borderWidth: 2,
                    borderRadius: 8,
                    hoverBackgroundColor: 'rgba(6, 182, 212, 0.7)',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                }
            }
        });
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Compass Rotation
    const windDirInput = document.getElementById('wind_direction');
    if (windDirInput) {
        windDirInput.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value) || 0;
            const compassWrap = document.getElementById('compassWrap');
            if (compassWrap) compassWrap.style.transform = `translateY(-50%) rotate(${val}deg)`;
        });
    }

    // Turbine Animation Speed
    const theoreticalPowerInput = document.getElementById('theoretical_power');
    if (theoreticalPowerInput) {
        theoreticalPowerInput.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value) || 0;
            const turbine = document.getElementById('turbineIcon');
            if (turbine) {
                if (val > 0) {
                    const duration = Math.max(0.2, 5 - (val / 500));
                    turbine.style.animationDuration = `${duration}s`;
                    turbine.style.animationPlayState = 'running';
                } else {
                    turbine.style.animationPlayState = 'paused';
                }
            }
        });
    }

    // Geolocation & Weather Integration
    const geoBtn = document.getElementById('geoBtn');
    if (geoBtn) {
        geoBtn.addEventListener('click', () => {
            const originalHtml = geoBtn.innerHTML;
            if (!navigator.geolocation) {
                alert("Geolocation is not supported by your browser.");
                return;
            }
            geoBtn.innerHTML = '<div class="spinner" style="display:block; width:14px; height:14px; border-width:2px;"></div> Fetching...';
            geoBtn.style.opacity = '0.7';
            geoBtn.disabled = true;

            navigator.geolocation.getCurrentPosition(async (position) => {
                const { latitude, longitude } = position.coords;
                try {
                    const response = await fetch('/fetch_weather', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ lat: latitude, lon: longitude })
                    });
                    const data = await response.json();
                    if (data.error) throw new Error(data.error);
                    document.getElementById('wind_speed_ms').value = data.wind_speed;
                    document.getElementById('wind_direction').value = data.wind_deg;
                    document.getElementById('wind_direction').dispatchEvent(new Event('input'));
                    geoBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"></path></svg> ${data.location}`;
                    geoBtn.classList.add('success-state');
                    setTimeout(() => {
                        geoBtn.innerHTML = originalHtml;
                        geoBtn.style.opacity = '1';
                        geoBtn.disabled = false;
                        geoBtn.classList.remove('success-state');
                    }, 4000);
                } catch (error) {
                    console.error("Weather fetch error:", error);
                    alert(`Failed to fetch weather: ${error.message}`);
                    geoBtn.innerHTML = originalHtml;
                    geoBtn.style.opacity = '1';
                    geoBtn.disabled = false;
                }
            }, (error) => {
                let msg = "Geolocation error.";
                if (error.code === error.PERMISSION_DENIED) msg = "Location permission denied.";
                alert(msg);
                geoBtn.innerHTML = originalHtml;
                geoBtn.style.opacity = '1';
                geoBtn.disabled = false;
            }, { timeout: 10000 });
        });
    }
});
