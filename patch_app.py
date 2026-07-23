import re

with open("static/app.js", "r") as f:
    content = f.read()

# 1. Add model fetching logic to populate pills
pills_logic = """
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
    };
"""

# Insert pills_logic right before `function loadPredictionData()`
content = content.replace("async function loadPredictionData() {", pills_logic + "\n    async function loadPredictionData() {")

# 2. Modify loadPredictionData to append query params
new_fetch = """
            // Gather selected models
            const params = new URLSearchParams();
            selectedModelsToCompare.forEach(m => params.append('models', m));
            
            const url = `${API.DATA}?${params.toString()}`;
            const resp = await fetch(url);
"""
content = re.sub(r'const resp = await fetch\(API\.DATA\);', new_fetch.strip(), content)

# 3. Modify renderPredictionData
old_render_prediction = """    function renderPredictionData(data) {
        // Update metrics
        setText('metric-ticker', data.ticker || '—');
        setText('metric-datapoints', data.data_points?.toLocaleString() || '—');
        setText('metric-mae', data.metrics?.MAE?.toFixed(4) || '—');

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

        // Error metrics
        if (data.metrics) {
            setText('metric-mae-detail', data.metrics.MAE?.toFixed(4) || '—');
            setText('metric-rmse', data.metrics.RMSE?.toFixed(4) || '—');
            setText('metric-mape', (data.metrics.MAPE || 0).toFixed(2) + '%');
        }

        // Render chart
        renderPriceChart(data);
    }"""

new_render_prediction = """    function renderPredictionData(data) {
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
    }"""

content = content.replace(old_render_prediction, new_render_prediction)

# 4. Modify renderPriceChart to plot multiple datasets
old_chart_datasets = """        const datasets = [
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
            },
            {
                label: 'Predicted Price',
                data: data.predicted_prices,
                borderColor: CHART_COLORS.predicted,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                borderDash: [6, 3],
                tension: 0, // Optimize: Disable bezier curves
                yAxisID: 'y',
                order: 2,
            },
        ];"""

new_chart_datasets = """        const datasets = [
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
        }"""

content = content.replace(old_chart_datasets, new_chart_datasets)

# 5. Call loadModelPills() in initDashboard()
content = content.replace("function initDashboard() {", "function initDashboard() {\n        loadModelPills();")

# 6. Call loadModelPills() after a model is uploaded or deleted so the pills stay updated
content = content.replace("loadModels();", "loadModels(); loadModelPills();")


with open("static/app.js", "w") as f:
    f.write(content)

print("Patched app.js successfully!")
