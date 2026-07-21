/**
 * Quaternion AI Predictor — Dashboard Application
 * 
 * SPA routing, Chart.js charts with zoom/pan, real-time training polling,
 * model management, and config form handling. All vendor libraries are local.
 */

(function () {
    'use strict';

    // ═══════════════════════════════════════════════
    // Constants
    // ═══════════════════════════════════════════════
    const API = {
        DATA: '/api/data',
        CONFIG: '/api/config',
        SEARCH: '/api/search',
        TRAIN: '/api/train',
        TRAINING_STATUS: '/api/training_status',
        MODELS: '/api/models',
        UPLOAD: '/api/upload_model',
        ACTIVATE: '/api/activate_model',
    };

    const POLL_INTERVAL = 500; // ms
    const SEARCH_DEBOUNCE = 300; // ms

    // Chart.js dark theme defaults
    const CHART_COLORS = {
        actual: '#00d4ff',
        predicted: '#7c3aed',
        signal: '#f59e0b',
        loss: '#ef4444',
        valLoss: '#f59e0b',
        mae: '#22c55e',
        valMae: '#3b82f6',
        grid: 'rgba(255, 255, 255, 0.04)',
        text: '#8892a8',
    };

    // ═══════════════════════════════════════════════
    // State
    // ═══════════════════════════════════════════════
    let trainingPollId = null;
    let priceChart = null;
    let lossChart = null;
    let maeChart = null;
    let searchTimeout = null;
    let currentConfig = null;

    // Drawing State
    let isDrawingMode = false;
    let isDrawing = false;
    let trendLines = [];
    let currentLine = null;

    // ═══════════════════════════════════════════════
    // DOM Ready
    // ═══════════════════════════════════════════════
    document.addEventListener('DOMContentLoaded', () => {
        initNavigation();
        initDashboard();
        initModels();
        initTraining();
        initConfig();

        // Load initial data
        loadConfig();
        loadModels(); loadModelPills();
    });

    // ═══════════════════════════════════════════════
    // Navigation (SPA Routing)
    // ═══════════════════════════════════════════════
    function initNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const sections = document.querySelectorAll('.section');

        // Make all sections visible
        sections.forEach(s => {
            s.style.display = 'block';
            s.classList.add('active');
        });

        // Click nav to scroll
        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const target = item.dataset.section;
                const section = document.getElementById(`section-${target}`);
                
                if (section) {
                    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }

                // Section-specific load actions
                if (target === 'models') loadModels(); loadModelPills();
                if (target === 'config') loadConfig();
            });
        });

        // Update active nav based on scroll position
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.id.replace('section-', '');
                    navItems.forEach(n => {
                        n.classList.toggle('active', n.dataset.section === id);
                    });
                }
            });
        }, {
            root: document.querySelector('.main-content'),
            threshold: 0.15
        });

        sections.forEach(s => observer.observe(s));
    }

    // ═══════════════════════════════════════════════
    // Dashboard Section
    // ═══════════════════════════════════════════════
    function initDashboard() {
        loadModelPills();
        const btnLoad = document.getElementById('btn-load-data');
        if (btnLoad) btnLoad.addEventListener('click', loadPredictionData);

        const btnResetZoom = document.getElementById('btn-reset-zoom');
        if (btnResetZoom) btnResetZoom.addEventListener('click', () => {
            if (priceChart) priceChart.resetZoom();
        });

        // Toggle buttons
        ['actual', 'predicted', 'signal'].forEach((key, idx) => {
            const btn = document.getElementById(`btn-toggle-${key}`);
            if (btn) {
                btn.addEventListener('click', () => {
                    if (!priceChart) return;
                    const meta = priceChart.getDatasetMeta(idx);
                    meta.hidden = !meta.hidden;
                    btn.classList.toggle('btn-active', !meta.hidden);
                    priceChart.update();
                });
            }
        });

        // Drawing Tools
        const btnToggleDraw = document.getElementById('btn-toggle-draw');
        const btnClearDraw = document.getElementById('btn-clear-draw');
        const canvas = document.getElementById('price-chart');

        if (btnToggleDraw) {
            btnToggleDraw.addEventListener('click', () => {
                isDrawingMode = !isDrawingMode;
                btnToggleDraw.classList.toggle('btn-active', isDrawingMode);
                if (btnClearDraw) btnClearDraw.style.display = isDrawingMode ? 'inline-block' : 'none';
                
                if (canvas) {
                    canvas.style.cursor = isDrawingMode ? 'crosshair' : 'default';
                }
                
                if (priceChart) {
                    // Disable panning when drawing
                    priceChart.options.plugins.zoom.pan.enabled = !isDrawingMode;
                    priceChart.update('none');
                }
            });
        }

        if (btnClearDraw) {
            btnClearDraw.addEventListener('click', () => {
                trendLines = [];
                if (priceChart) priceChart.update('none');
            });
        }

        if (canvas) {
            canvas.addEventListener('mousedown', (e) => {
                if (!isDrawingMode || !priceChart) return;
                isDrawing = true;
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                const xVal = priceChart.scales.x.getValueForPixel(x);
                const yVal = priceChart.scales.y.getValueForPixel(y);
                
                currentLine = { start: {x: xVal, y: yVal}, end: {x: xVal, y: yVal} };
            });
            
            canvas.addEventListener('mousemove', (e) => {
                if (!isDrawing || !currentLine || !priceChart) return;
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                currentLine.end = {
                    x: priceChart.scales.x.getValueForPixel(x),
                    y: priceChart.scales.y.getValueForPixel(y)
                };
                priceChart.update('none'); // Update to preview the line
            });
            
            const stopDrawing = () => {
                if (isDrawing && currentLine) {
                    trendLines.push(currentLine);
                    currentLine = null;
                    isDrawing = false;
                    if (priceChart) priceChart.update('none');
                }
            };
            
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('mouseout', stopDrawing);
        }
    }

    
    let availableModels = [];
    let selectedModelsToCompare = new Set();
    const PRESET_COLORS = ['#7c3aed', '#ec4899', '#f59e0b', '#10b981', '#3b82f6'];

    async function loadModelPills() {
        try {
            const resp = await fetch(API.MODELS);
            const data = await resp.json();
            if (!data.models) return;
            
            availableModels = data.models.map(m => m.filename);
            
            // By default, select the primary active model
            if (selectedModelsToCompare.size === 0 && data.active?.primary) {
                selectedModelsToCompare.add(data.active.primary);
            }
            
            renderModelPills();
        } catch (e) {
            console.error("Failed to load models for pills", e);
        }
    }

    function renderModelPills() {
        const container = document.getElementById('model-pills-container');
        if (!container) return;
        
        container.innerHTML = availableModels.map((filename, idx) => {
            const isSelected = selectedModelsToCompare.has(filename);
            const color = PRESET_COLORS[idx % PRESET_COLORS.length];
            return `
                <div class="model-pill ${isSelected ? 'selected' : ''}" data-filename="${escapeHtml(filename)}" onclick="window._toggleModelPill('${escapeHtml(filename)}')">
                    <div class="model-pill-color-indicator" style="background: ${isSelected ? color : ''}"></div>
                    ${escapeHtml(filename)}
                </div>
            `;
        }).join('');
    }

    window._toggleModelPill = function(filename) {
        if (selectedModelsToCompare.has(filename)) {
            // Require at least one model selected
            if (selectedModelsToCompare.size > 1) {
                selectedModelsToCompare.delete(filename);
            } else {
                showNotification("At least one model must be selected", "warning");
                return;
            }
        } else {
            selectedModelsToCompare.add(filename);
        }
        renderModelPills();
        loadPredictionData();
    };

    async function loadPredictionData() {
        const btn = document.getElementById('btn-load-data');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Loading...';
        }

        setStatus('loading', 'Loading data...');

        try {
            // Gather selected models
            const params = new URLSearchParams();
            selectedModelsToCompare.forEach(m => params.append('models', m));
            
            const url = `${API.DATA}?${params.toString()}`;
            const resp = await fetch(url);
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.error || `HTTP ${resp.status}`);
            }

            const data = await resp.json();
            renderPredictionData(data);
            setStatus('online', 'Data loaded');
        } catch (e) {
            console.error('Failed to load data:', e);
            setStatus('offline', `Error: ${e.message}`);
            showNotification(`Failed to load data: ${e.message}`, 'danger');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span>◆</span> Load Prediction Data';
            }
        }
    }

    function renderPredictionData(data) {
        // Update basic metrics
        setText('metric-ticker', data.ticker || '—');
        setText('metric-datapoints', data.data_points?.toLocaleString() || '—');
        
        // Show MAE for the first plotted model in the small card
        const primaryModel = data.model_predictions?.[0];
        setText('metric-mae', primaryModel?.metrics?.MAE?.toFixed(4) || '—');

        const signal = data.next_signal || 0;
        setText('metric-signal', signal.toFixed(4));

        // Signal badge
        const badgeContainer = document.getElementById('signal-badge-container');
        if (badgeContainer) {
            let badgeClass, badgeText;
            if (signal > 0.5) {
                badgeClass = 'signal-sell';
                badgeText = 'MOMENTUM CREST (SELL)';
            } else if (signal < -0.5) {
                badgeClass = 'signal-buy';
                badgeText = 'MOMENTUM TROUGH (BUY)';
            } else {
                badgeClass = 'signal-hold';
                badgeText = 'STABLE MOMENTUM (HOLD)';
            }
            badgeContainer.innerHTML = `<span class="signal-badge ${badgeClass}">${badgeText}</span>`;
        }

        // Populate Comparative Metrics Table
        const tbody = document.getElementById('metrics-table-body');
        if (tbody && data.model_predictions) {
            tbody.innerHTML = data.model_predictions.map((mod, idx) => {
                const color = PRESET_COLORS[idx % PRESET_COLORS.length];
                return `
                    <tr>
                        <td>
                            <div class="metric-name-cell">
                                <div class="model-pill-color-indicator" style="background: ${color}"></div>
                                ${escapeHtml(mod.name)}
                            </div>
                        </td>
                        <td style="font-family: monospace;">${mod.metrics.MAE?.toFixed(4)}</td>
                        <td style="font-family: monospace;">${mod.metrics.RMSE?.toFixed(4)}</td>
                        <td style="font-family: monospace;">${(mod.metrics.MAPE || 0).toFixed(2)}%</td>
                    </tr>
                `;
            }).join('');
        }

        // Render chart
        renderPriceChart(data);
    }

    function renderPriceChart(data) {
        const ctx = document.getElementById('price-chart');
        if (!ctx) return;

        if (priceChart) priceChart.destroy();

        const datasets = [
            {
                label: 'Actual Price',
                data: data.actual_prices,
                borderColor: CHART_COLORS.actual,
                backgroundColor: hexToRgba(CHART_COLORS.actual, 0.05),
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: true,
                tension: 0, // Optimize: Disable bezier curves for performance
                yAxisID: 'y',
                order: 1,
            }
        ];
        
        if (data.model_predictions) {
            data.model_predictions.forEach((mod, idx) => {
                const color = PRESET_COLORS[idx % PRESET_COLORS.length];
                datasets.push({
                    label: `Predicted (${mod.name})`,
                    data: mod.predicted_prices,
                    borderColor: color,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    borderDash: [6, 3],
                    tension: 0,
                    yAxisID: 'y',
                    order: 2 + idx,
                });
            });
        }

        // Add signal dataset if available
        if (data.signals && data.signals.length > 0) {
            // Pad signals to align with the end of the price data
            const paddedSignals = new Array(data.actual_prices.length - data.signals.length).fill(null)
                .concat(data.signals);
            datasets.push({
                label: 'Oscillator Signal',
                data: paddedSignals,
                borderColor: CHART_COLORS.signal,
                backgroundColor: hexToRgba(CHART_COLORS.signal, 0.08),
                borderWidth: 1.5,
                pointRadius: 0,
                fill: true,
                tension: 0, // Optimize: Disable bezier curves
                yAxisID: 'y1',
                order: 3,
            });
        }

        const signalHighlightsPlugin = {
            id: 'signalHighlights',
            beforeDraw(chart, args, options) {
                const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
                const signalDataset = chart.data.datasets.find(d => d.label === 'Oscillator Signal');
                if (!signalDataset) return;
                
                const signals = signalDataset.data;
                ctx.save();
                
                for (let i = 0; i < signals.length; i++) {
                    const signal = signals[i];
                    if (signal === null || signal === undefined) continue;
                    
                    if (Math.abs(signal) > 0.05) {
                        const isSell = signal > 0;
                        // Max opacity 0.15 for subtle background
                        const alpha = Math.min(Math.abs(signal) * 0.15, 0.15); 
                        // Red for >0 (Sell / Upward Divergence), Green for <0 (Buy / Downward Divergence)
                        ctx.fillStyle = isSell ? `rgba(239, 68, 68, ${alpha})` : `rgba(34, 197, 94, ${alpha})`;
                        
                        let startX = x.getPixelForValue(i);
                        let width = 0;
                        
                        if (i < signals.length - 1) {
                            width = x.getPixelForValue(i + 1) - startX;
                        } else if (i > 0) {
                            width = startX - x.getPixelForValue(i - 1);
                        }
                        
                        if (width > 0) {
                            // Draw centered on the data point
                            ctx.fillRect(startX - width/2, top, width, bottom - top);
                        }
                    }
                }
                ctx.restore();
            }
        };

        const trendlinePlugin = {
            id: 'trendlines',
            afterDraw(chart, args, options) {
                const { ctx, scales: { x, y }, chartArea: { top, bottom, left, right } } = chart;
                
                ctx.save();
                
                // Set clip to chart area so lines don't draw over axes
                ctx.beginPath();
                ctx.rect(left, top, right - left, bottom - top);
                ctx.clip();
                
                // Helper to draw a line
                const drawLine = (line, color, dash) => {
                    const pxStartX = x.getPixelForValue(line.start.x);
                    const pxStartY = y.getPixelForValue(line.start.y);
                    const pxEndX = x.getPixelForValue(line.end.x);
                    const pxEndY = y.getPixelForValue(line.end.y);
                    
                    ctx.beginPath();
                    ctx.moveTo(pxStartX, pxStartY);
                    ctx.lineTo(pxEndX, pxEndY);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    if (dash) ctx.setLineDash(dash);
                    else ctx.setLineDash([]);
                    ctx.stroke();
                    
                    // Draw handles
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(pxStartX, pxStartY, 4, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.beginPath();
                    ctx.arc(pxEndX, pxEndY, 4, 0, Math.PI * 2);
                    ctx.fill();
                };
                
                // Draw all saved lines
                trendLines.forEach(line => {
                    drawLine(line, 'rgba(0, 212, 255, 0.8)'); // Match actual price color
                });
                
                // Draw current preview line
                if (isDrawing && currentLine) {
                    drawLine(currentLine, 'rgba(0, 212, 255, 1)', [5, 5]); // Dashed preview
                }
                
                ctx.restore();
            }
        };

        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: datasets,
            },
            plugins: [signalHighlightsPlugin, trendlinePlugin],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,      // Optimize: disable animations for large datasets
                normalized: true,      // Optimize: bypass data normalization
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: CHART_COLORS.text,
                            font: { family: "'Outfit', sans-serif", size: 12 },
                            padding: 16,
                            usePointStyle: true,
                            pointStyle: 'rectRounded',
                        },
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 14, 26, 0.95)',
                        titleColor: '#e8edf5',
                        bodyColor: '#8892a8',
                        borderColor: 'rgba(255,255,255,0.08)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        padding: 12,
                        titleFont: { family: "'Outfit', sans-serif", weight: '600' },
                        bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
                        displayColors: true,
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x',
                            modifierKey: null,
                        },
                        zoom: {
                            wheel: { enabled: true, speed: 0.05 },
                            pinch: { enabled: true },
                            drag: {
                                enabled: false,
                            },
                            mode: 'x',
                        },
                    },
                },
                scales: {
                    x: {
                        display: true,
                        ticks: {
                            color: CHART_COLORS.text,
                            font: { size: 10, family: "'JetBrains Mono', monospace" },
                            maxTicksLimit: 12,
                            maxRotation: 45,
                        },
                        grid: { color: CHART_COLORS.grid },
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Price',
                            color: CHART_COLORS.text,
                            font: { size: 12, family: "'Outfit', sans-serif" },
                        },
                        ticks: {
                            color: CHART_COLORS.text,
                            font: { size: 10, family: "'JetBrains Mono', monospace" },
                        },
                        grid: { color: CHART_COLORS.grid },
                    },
                    y1: {
                        type: 'linear',
                        display: data.signals && data.signals.length > 0,
                        position: 'right',
                        min: -1.2,
                        max: 1.2,
                        title: {
                            display: true,
                            text: 'Signal',
                            color: CHART_COLORS.signal,
                            font: { size: 12, family: "'Outfit', sans-serif" },
                        },
                        ticks: {
                            color: CHART_COLORS.signal,
                            font: { size: 10, family: "'JetBrains Mono', monospace" },
                            callback: v => v.toFixed(1),
                        },
                        grid: { drawOnChartArea: false },
                    },
                },
            },
        });

        // Initialize toggle button states
        ['actual', 'predicted', 'signal'].forEach((key) => {
            const btn = document.getElementById(`btn-toggle-${key}`);
            if (btn) btn.classList.add('btn-active');
        });
    }

    // ═══════════════════════════════════════════════
    // Models Section
    // ═══════════════════════════════════════════════
    function initModels() {
        // Upload zones
        ['primary', 'oscillator'].forEach(type => {
            const zone = document.getElementById(`upload-zone-${type}`);
            const input = document.getElementById(`upload-input-${type}`);
            if (!zone || !input) return;

            zone.addEventListener('click', () => input.click());
            zone.addEventListener('dragover', e => {
                e.preventDefault();
                zone.classList.add('drag-over');
            });
            zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
            zone.addEventListener('drop', e => {
                e.preventDefault();
                zone.classList.remove('drag-over');
                if (e.dataTransfer.files.length) {
                    uploadModel(e.dataTransfer.files[0], type);
                }
            });
            input.addEventListener('change', () => {
                if (input.files.length) {
                    uploadModel(input.files[0], type);
                    input.value = '';
                }
            });
        });

        const btnRefresh = document.getElementById('btn-refresh-models');
        if (btnRefresh) btnRefresh.addEventListener('click', loadModels);
    }

    async function loadModels() {
        const container = document.getElementById('models-list');
        if (!container) return;

        try {
            const resp = await fetch(API.MODELS);
            const data = await resp.json();

            if (!data.models || data.models.length === 0) {
                container.innerHTML = '<p class="text-muted" style="padding: 20px; text-align: center;">No saved models found. Train a model or upload one above.</p>';
                return;
            }

            container.innerHTML = data.models.map(model => `
                <div class="model-card ${model.is_active_primary || model.is_active_oscillator ? 'active' : ''}">
                    <div class="model-card-main">
                        <div class="model-card-info">
                            <div class="model-name">${escapeHtml(model.filename)}</div>
                            <div class="model-meta">
                                ${model.size_human} · Modified ${formatDate(model.modified)}
                                ${model.is_active_primary ? '<span class="card-badge badge-cyan">Active Primary</span>' : ''}
                                ${model.is_active_oscillator ? '<span class="card-badge badge-purple">Active Oscillator</span>' : ''}
                            </div>
                        </div>
                        <div class="model-actions">
                            ${!model.is_active_primary ? `<button class="btn btn-sm btn-secondary" onclick="window._activateModel('${escapeHtml(model.filename)}', 'primary')">Set Primary</button>` : ''}
                            ${!model.is_active_oscillator ? `<button class="btn btn-sm btn-secondary" onclick="window._activateModel('${escapeHtml(model.filename)}', 'oscillator')">Set Oscillator</button>` : ''}
                            <a class="btn btn-sm btn-secondary" href="/api/models/${encodeURIComponent(model.filename)}/download" download>⬇ Download</a>
                            ${!model.is_active_primary && !model.is_active_oscillator ? `<button class="btn btn-sm btn-danger" onclick="window._deleteModel('${escapeHtml(model.filename)}')">✕</button>` : ''}
                        </div>
                    </div>
                </div>
            `).join('');

        } catch (e) {
            container.innerHTML = `<p class="text-muted" style="padding: 20px;">Error loading models: ${escapeHtml(e.message)}</p>`;
        }
    }

    async function uploadModel(file, type) {
        if (!file.name.endsWith('.keras')) {
            showNotification('Only .keras files are accepted', 'danger');
            return;
        }

        showNotification(`Uploading ${file.name}...`, 'info');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch(`${API.UPLOAD}?model_type=${type}`, {
                method: 'POST',
                body: formData,
            });
            const data = await resp.json();
            if (resp.ok) {
                showNotification(`Model uploaded: ${data.filename}`, 'success');
                loadModels(); loadModelPills();
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (e) {
            showNotification(`Upload failed: ${e.message}`, 'danger');
        }
    }

    // Expose to onclick handlers in HTML
    window._activateModel = async function (filename, type) {
        try {
            const resp = await fetch(`${API.ACTIVATE}?filename=${encodeURIComponent(filename)}&model_type=${type}`, {
                method: 'POST',
            });
            if (resp.ok) {
                showNotification(`${filename} set as active ${type} model`, 'success');
                loadModels(); loadModelPills();
            } else {
                const err = await resp.json();
                throw new Error(err.detail || 'Activation failed');
            }
        } catch (e) {
            showNotification(`Failed: ${e.message}`, 'danger');
        }
    };

    window._deleteModel = async function (filename) {
        if (!confirm(`Delete model "${filename}"? This cannot be undone.`)) return;
        try {
            const resp = await fetch(`/api/models/${encodeURIComponent(filename)}`, { method: 'DELETE' });
            if (resp.ok) {
                showNotification(`Deleted: ${filename}`, 'success');
                loadModels(); loadModelPills();
            } else {
                const err = await resp.json();
                throw new Error(err.detail || 'Delete failed');
            }
        } catch (e) {
            showNotification(`Failed: ${e.message}`, 'danger');
        }
    };

    // ═══════════════════════════════════════════════
    // Training Section
    // ═══════════════════════════════════════════════
    function initTraining() {
        const btnStart = document.getElementById('btn-start-training');
        if (btnStart) btnStart.addEventListener('click', startTraining);

        const btnClear = document.getElementById('btn-clear-log');
        if (btnClear) btnClear.addEventListener('click', () => {
            const body = document.getElementById('terminal-body');
            if (body) body.innerHTML = '<div class="terminal-line">[SYSTEM] Log cleared.</div>';
        });

        // Initialize training charts
        initTrainingCharts();

        // Check if training is already in progress
        checkTrainingStatus();
    }

    function initTrainingCharts() {
        const lossCtx = document.getElementById('loss-chart');
        const maeCtx = document.getElementById('mae-chart');

        const chartOptions = (label, color1, color2) => ({
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: `Train ${label}`,
                        data: [],
                        borderColor: color1,
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 3,
                        pointBackgroundColor: color1,
                        tension: 0.3,
                    },
                    {
                        label: `Val ${label}`,
                        data: [],
                        borderColor: color2,
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 3,
                        pointBackgroundColor: color2,
                        borderDash: [5, 3],
                        tension: 0.3,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: CHART_COLORS.text,
                            font: { family: "'Outfit', sans-serif", size: 11 },
                            usePointStyle: true,
                        },
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 14, 26, 0.95)',
                        titleColor: '#e8edf5',
                        bodyColor: '#8892a8',
                        borderColor: 'rgba(255,255,255,0.08)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
                    },
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Epoch', color: CHART_COLORS.text, font: { size: 11 } },
                        ticks: { color: CHART_COLORS.text, font: { size: 10 } },
                        grid: { color: CHART_COLORS.grid },
                    },
                    y: {
                        title: { display: true, text: label, color: CHART_COLORS.text, font: { size: 11 } },
                        ticks: { color: CHART_COLORS.text, font: { size: 10 } },
                        grid: { color: CHART_COLORS.grid },
                    },
                },
            },
        });

        if (lossCtx) lossChart = new Chart(lossCtx, chartOptions('Loss', CHART_COLORS.loss, CHART_COLORS.valLoss));
        if (maeCtx) maeChart = new Chart(maeCtx, chartOptions('MAE', CHART_COLORS.mae, CHART_COLORS.valMae));
    }

    async function startTraining() {
        const btn = document.getElementById('btn-start-training');
        const epochs = parseInt(document.getElementById('train-epochs')?.value) || 20;
        const oscEpochs = parseInt(document.getElementById('train-osc-epochs')?.value) || 20;
        const modelName = document.getElementById('train-model-name')?.value.trim() || 'best_model';

        try {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Starting...';

            const resp = await fetch(`${API.TRAIN}?epochs=${epochs}&oscillator_epochs=${oscEpochs}&model_name=${encodeURIComponent(modelName)}`, {
                method: 'POST',
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }

            showNotification('Training started!', 'success');
            setStatus('training', 'Training in progress');

            // Show progress bar
            const progressContainer = document.getElementById('training-progress-container');
            if (progressContainer) progressContainer.style.display = 'block';

            // Start polling
            startTrainingPoll();

        } catch (e) {
            showNotification(`Failed to start training: ${e.message}`, 'danger');
            btn.disabled = false;
            btn.innerHTML = '⚡ Start Training';
        }
    }

    function startTrainingPoll() {
        if (trainingPollId) clearInterval(trainingPollId);
        trainingPollId = setInterval(pollTrainingStatus, POLL_INTERVAL);
    }

    async function pollTrainingStatus() {
        try {
            const resp = await fetch(API.TRAINING_STATUS);
            const state = await resp.json();
            updateTrainingUI(state);

            if (!state.is_training && state.phase !== 'idle') {
                clearInterval(trainingPollId);
                trainingPollId = null;

                const btn = document.getElementById('btn-start-training');
                if (btn) {
                    btn.disabled = false;
                    btn.innerHTML = '⚡ Start Training';
                }

                if (state.phase === 'complete') {
                    setStatus('online', 'Training complete');
                    showNotification('Training complete! Models saved.', 'success');
                } else if (state.phase === 'error') {
                    setStatus('offline', 'Training failed');
                    showNotification(`Training error: ${state.error}`, 'danger');
                }
            }
        } catch (e) {
            console.error('Training poll error:', e);
        }
    }

    async function checkTrainingStatus() {
        try {
            const resp = await fetch(API.TRAINING_STATUS);
            const state = await resp.json();
            if (state.is_training) {
                setStatus('training', 'Training in progress');
                const btn = document.getElementById('btn-start-training');
                if (btn) {
                    btn.disabled = true;
                    btn.innerHTML = '<span class="spinner"></span> Training...';
                }
                const progressContainer = document.getElementById('training-progress-container');
                if (progressContainer) progressContainer.style.display = 'block';
                startTrainingPoll();
                updateTrainingUI(state);
            }
        } catch (e) {
            // Server not ready yet, ignore
        }
    }

    function updateTrainingUI(state) {
        // Phase badge
        const phaseBadge = document.getElementById('training-phase-badge');
        if (phaseBadge) {
            const phaseColors = {
                idle: 'badge-muted',
                primary: 'badge-cyan',
                oscillator: 'badge-purple',
                complete: 'badge-success',
                error: 'badge-danger',
            };
            const cls = phaseColors[state.phase] || 'badge-muted';
            phaseBadge.innerHTML = `<span class="card-badge ${cls}">${state.phase.toUpperCase()}</span>`;
        }

        // Progress bar
        if (state.total_epochs > 0) {
            const pct = (state.epoch / state.total_epochs) * 100;
            const fill = document.getElementById('training-progress-fill');
            const label = document.getElementById('training-progress-label');
            if (fill) fill.style.width = `${pct}%`;
            if (label) label.textContent = `Epoch ${state.epoch}/${state.total_epochs} — ${pct.toFixed(0)}%`;
        }

        // Training charts
        if (state.history) {
            const epochs = state.history.loss.map((_, i) => i + 1);

            if (lossChart) {
                lossChart.data.labels = epochs;
                lossChart.data.datasets[0].data = state.history.loss;
                lossChart.data.datasets[1].data = state.history.val_loss;
                lossChart.update('none');
            }

            if (maeChart) {
                maeChart.data.labels = epochs;
                maeChart.data.datasets[0].data = state.history.mae;
                maeChart.data.datasets[1].data = state.history.val_mae;
                maeChart.update('none');
            }
        }

        // Terminal log
        const body = document.getElementById('terminal-body');
        if (body && state.logs) {
            body.innerHTML = state.logs
                .map(line => `<div class="terminal-line">${escapeHtml(line)}</div>`)
                .join('');
            // Auto-scroll to bottom
            const terminal = document.getElementById('training-terminal');
            if (terminal) terminal.scrollTop = terminal.scrollHeight;
        }
    }

    // ═══════════════════════════════════════════════
    // Configuration Section
    // ═══════════════════════════════════════════════
    function initConfig() {
        // Collapsible sections
        document.querySelectorAll('.config-section-header').forEach(header => {
            header.addEventListener('click', () => {
                const sectionId = header.dataset.section;
                const section = document.getElementById(sectionId);
                if (section) section.classList.toggle('open');
            });
        });

        // Ticker autocomplete
        const tickerInput = document.getElementById('cfg-ticker-primary');
        if (tickerInput) {
            tickerInput.addEventListener('input', () => {
                clearTimeout(searchTimeout);
                const q = tickerInput.value.trim();
                if (q.length < 1) {
                    hideDropdown();
                    return;
                }
                searchTimeout = setTimeout(() => searchTicker(q), SEARCH_DEBOUNCE);
            });

            // Close dropdown on outside click
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.autocomplete-wrapper')) hideDropdown();
            });
        }

        // Config form submits
        const formDataSource = document.getElementById('form-data-source');
        if (formDataSource) {
            formDataSource.addEventListener('submit', async (e) => {
                e.preventDefault();
                await saveFormConfig('form-data-source', 'btn-save-data-source', '💾 Save Data Source');
            });
        }

        const formHyperparams = document.getElementById('form-hyperparameters');
        if (formHyperparams) {
            formHyperparams.addEventListener('submit', async (e) => {
                e.preventDefault();
                await saveFormConfig('form-hyperparameters', 'btn-save-hyperparameters', '💾 Save Hyperparameters');
            });
        }

        // Reset button
        const btnReset = document.getElementById('btn-reset-hyperparameters');
        if (btnReset) {
            btnReset.addEventListener('click', () => {
                if (currentConfig) populateConfigForm(currentConfig);
            });
        }
    }

    async function loadConfig() {
        try {
            const resp = await fetch(API.CONFIG);
            currentConfig = await resp.json();
            populateConfigForm(currentConfig);
        } catch (e) {
            console.error('Failed to load config:', e);
        }
    }

    function populateConfigForm(cfg) {
        // Data source
        setInputValue('cfg-ticker-primary', cfg.data_source?.ticker_primary);
        setInputValue('cfg-ticker-fallback', cfg.data_source?.ticker_fallback);
        setInputValue('cfg-start-date', cfg.data_source?.default_start_date);
        setInputValue('cfg-interval', cfg.data_source?.default_interval);

        // Sequence
        setInputValue('cfg-seq-length', cfg.sequence?.sequence_length);

        // Dual-stream
        setCheckbox('cfg-dual-stream', cfg.dual_stream?.dual_stream);
        setInputValue('cfg-ctx-lstm-units', cfg.dual_stream?.context_lstm_units);
        setInputValue('cfg-volume-ma', cfg.dual_stream?.volume_ma_window);
        setInputValue('cfg-fusion', cfg.dual_stream?.fusion_strategy);
        setInputValue('cfg-fusion-dense', cfg.dual_stream?.fusion_dense_units);
        setInputValue('cfg-ctx-dropout', cfg.dual_stream?.context_dropout_rate);

        // Model architecture
        setInputValue('cfg-model-type', cfg.model_architecture?.model_type);
        setInputValue('cfg-lstm-units', cfg.model_architecture?.lstm_units);
        setInputValue('cfg-dense-units', cfg.model_architecture?.dense_units);
        setInputValue('cfg-dropout', cfg.model_architecture?.dropout_rate);

        // Training
        setInputValue('cfg-lr', cfg.training?.learning_rate);
        setInputValue('cfg-batch', cfg.training?.batch_size);
        setInputValue('cfg-epochs-config', cfg.training?.epochs);
        setInputValue('cfg-val-split', cfg.training?.validation_split);
        setInputValue('cfg-train-split', cfg.training?.train_test_split);
        setInputValue('cfg-seed', cfg.reproducibility?.random_seed);

        // Oscillator
        setInputValue('cfg-osc-seq', cfg.oscillator?.oscillator_seq_len);
        setInputValue('cfg-osc-lstm', cfg.oscillator?.oscillator_lstm_units);
        setInputValue('cfg-osc-dense', cfg.oscillator?.oscillator_dense_units);
        setInputValue('cfg-osc-epochs-config', cfg.oscillator?.oscillator_epochs);
        setInputValue('cfg-osc-lr', cfg.oscillator?.oscillator_learning_rate);
        setCheckbox('cfg-mean-center', cfg.oscillator?.mean_center_oscillator);

        // Update training section defaults too
        setInputValue('train-epochs', cfg.training?.epochs);
        setInputValue('train-osc-epochs', cfg.oscillator?.oscillator_epochs);
    }

    async function saveFormConfig(formId, btnId, originalHtml) {
        const btn = document.getElementById(btnId);
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Saving...';

        // Build update object from form
        const update = {};
        const form = document.getElementById(formId);

        form.querySelectorAll('input[name], select[name]').forEach(el => {
            const name = el.name;
            if (!name) return;

            if (el.type === 'checkbox') {
                update[name] = el.checked;
            } else if (el.type === 'number') {
                const v = parseFloat(el.value);
                if (!isNaN(v)) update[name] = v;
            } else {
                if (el.value.trim()) update[name] = el.value.trim();
            }
        });

        // Ensure integers are integers
        const intFields = [
            'sequence_length', 'context_lstm_units', 'volume_ma_window',
            'fusion_dense_units', 'lstm_units', 'dense_units', 'batch_size',
            'epochs', 'oscillator_seq_len', 'oscillator_lstm_units',
            'oscillator_dense_units', 'oscillator_epochs', 'random_seed',
        ];
        intFields.forEach(f => {
            if (update[f] !== undefined) update[f] = Math.round(update[f]);
        });

        try {
            const resp = await fetch(API.CONFIG, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(update),
            });

            if (resp.ok) {
                showNotification('Configuration saved!', 'success');
                await loadConfig(); // Refresh from server
            } else {
                const err = await resp.json();
                throw new Error(err.detail || 'Save failed');
            }
        } catch (e) {
            showNotification(`Failed to save: ${e.message}`, 'danger');
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalHtml;
        }
    }

    // Ticker search
    async function searchTicker(query) {
        try {
            const resp = await fetch(`${API.SEARCH}?q=${encodeURIComponent(query)}`);
            const data = await resp.json();

            const dropdown = document.getElementById('ticker-dropdown');
            if (!dropdown) return;

            if (!data.results || data.results.length === 0) {
                hideDropdown();
                return;
            }

            dropdown.innerHTML = data.results.map(r => `
                <div class="autocomplete-item" data-symbol="${escapeHtml(r.symbol)}">
                    <span class="autocomplete-symbol">${escapeHtml(r.symbol)}</span>
                    <span class="autocomplete-name">${escapeHtml(r.shortname || '')} · ${escapeHtml(r.exchange || '')}</span>
                </div>
            `).join('');

            dropdown.style.display = 'block';

            // Click handlers
            dropdown.querySelectorAll('.autocomplete-item').forEach(item => {
                item.addEventListener('click', () => {
                    document.getElementById('cfg-ticker-primary').value = item.dataset.symbol;
                    hideDropdown();
                });
            });

        } catch (e) {
            console.error('Search error:', e);
        }
    }

    function hideDropdown() {
        const dropdown = document.getElementById('ticker-dropdown');
        if (dropdown) dropdown.style.display = 'none';
    }

    // ═══════════════════════════════════════════════
    // Utilities
    // ═══════════════════════════════════════════════
    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function setInputValue(id, value) {
        const el = document.getElementById(id);
        if (el && value !== undefined && value !== null) {
            el.value = value;
        }
    }

    function setCheckbox(id, checked) {
        const el = document.getElementById(id);
        if (el) el.checked = !!checked;
    }

    function setStatus(state, text) {
        const dot = document.getElementById('status-dot');
        const label = document.getElementById('status-text');

        if (dot) {
            dot.className = 'status-dot';
            if (state === 'online' || state === 'loading') dot.classList.add('online');
            else if (state === 'training') dot.classList.add('training');
            else if (state === 'offline') dot.classList.add('offline');
        }
        if (label) label.textContent = text;
    }

    function escapeHtml(str) {
        const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
        return String(str).replace(/[&<>"']/g, m => map[m]);
    }

    function hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function formatDate(isoStr) {
        try {
            const d = new Date(isoStr);
            return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        } catch {
            return isoStr;
        }
    }

    // ═══════════════════════════════════════════════
    // Notifications
    // ═══════════════════════════════════════════════
    function showNotification(message, type = 'info') {
        // Create notification element
        let container = document.getElementById('notification-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notification-container';
            container.style.cssText = 'position:fixed;top:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px;';
            document.body.appendChild(container);
        }

        const notif = document.createElement('div');
        const colors = {
            success: { bg: 'rgba(34,197,94,0.15)', border: 'rgba(34,197,94,0.3)', color: '#22c55e' },
            danger: { bg: 'rgba(239,68,68,0.15)', border: 'rgba(239,68,68,0.3)', color: '#ef4444' },
            info: { bg: 'rgba(0,212,255,0.1)', border: 'rgba(0,212,255,0.2)', color: '#00d4ff' },
        };
        const c = colors[type] || colors.info;
        notif.style.cssText = `
            background:${c.bg}; border:1px solid ${c.border}; color:${c.color};
            padding:12px 20px; border-radius:10px; font-size:0.85rem; font-family:'Outfit',sans-serif;
            backdrop-filter:blur(16px); box-shadow:0 4px 24px rgba(0,0,0,0.3);
            animation: fadeIn 300ms ease-out; max-width: 400px; word-wrap: break-word;
        `;
        notif.textContent = message;
        container.appendChild(notif);

        setTimeout(() => {
            notif.style.opacity = '0';
            notif.style.transition = 'opacity 300ms';
            setTimeout(() => notif.remove(), 300);
        }, 4000);
    }

})();
