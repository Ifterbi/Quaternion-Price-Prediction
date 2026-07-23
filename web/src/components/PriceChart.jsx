import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  BarController,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Chart } from 'react-chartjs-2';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  BarController,
  Title,
  Tooltip,
  Legend,
  zoomPlugin
);

// Palette for different model predictions
const CHART_COLORS = [
  '#7c3aed', // Purple
  '#f59e0b', // Orange
  '#22c55e', // Green
  '#ef4444', // Red
  '#ec4899', // Pink
];

export default function PriceChart({ data, showSignal = true }) {
  if (!data || !data.dates || !data.actual_prices) return <p>No chart data available</p>;

  let signalValues = [];
  let signalType = 'residual';
  let buyThresh = null;
  let sellThresh = null;
  let pBuy = [];
  let pSell = [];

  const rawSignals = data.signals;
  if (Array.isArray(rawSignals)) {
      signalValues = rawSignals;
  } else if (rawSignals && rawSignals.type) {
      signalType = rawSignals.type;
      signalValues = rawSignals.values || [];
      if (signalType === 'threshold') {
          buyThresh = rawSignals.buy_threshold;
          sellThresh = rawSignals.sell_threshold;
      } else if (signalType === 'classification') {
          pBuy = rawSignals.p_buy || [];
          pSell = rawSignals.p_sell || [];
      }
  }

  let paddedSignals = null;
  let paddedPBuy = null;
  let paddedPSell = null;
  if (showSignal && signalValues && signalValues.length > 0) {
    let trimmedSignals = signalValues;
    let trimmedPBuy = pBuy;
    let trimmedPSell = pSell;
    
    if (signalValues.length > data.actual_prices.length) {
        const diff = signalValues.length - data.actual_prices.length;
        trimmedSignals = signalValues.slice(diff);
        if (signalType === 'classification') {
            trimmedPBuy = pBuy.slice(diff);
            trimmedPSell = pSell.slice(diff);
        }
    }
    
    const padLen = Math.max(0, data.actual_prices.length - trimmedSignals.length);
    const padArr = new Array(padLen).fill(null);
    paddedSignals = padArr.concat(trimmedSignals);
    if (signalType === 'classification') {
        paddedPBuy = padArr.concat(trimmedPBuy);
        paddedPSell = padArr.concat(trimmedPSell);
    }
  }

  const datasets = [];
  const chartDataLength = data.actual_prices.length;

  // Calculate stats for residual highlights if needed
  let resMean = 0;
  let resStd = 0;
  if (signalType === 'residual' && paddedSignals) {
      const validS = paddedSignals.filter(s => s !== null && s !== undefined);
      if (validS.length > 0) {
          resMean = validS.reduce((a, b) => a + b, 0) / validS.length;
          resStd = Math.sqrt(validS.reduce((a, b) => a + Math.pow(b - resMean, 2), 0) / validS.length);
      }
  }

  // Add Vertical Highlight Bar Dataset (this goes first to act as a background)
  if (paddedSignals) {
    datasets.push({
      type: 'bar',
      label: 'Signal Highlight',
      data: paddedSignals.map(s => (s !== null && s !== undefined) ? 1 : 0),
      backgroundColor: paddedSignals.map((s, idx) => {
        if (s === null || s === undefined) return 'transparent';
        if (signalType === 'classification') {
             const pb = paddedPBuy[idx];
             const ps = paddedPSell[idx];
             if (pb > ps && pb > 0.4) return 'rgba(34, 197, 94, 0.15)'; // Green
             if (ps > pb && ps > 0.4) return 'rgba(239, 68, 68, 0.15)'; // Red
             return 'rgba(234, 179, 8, 0.15)'; // Yellow
        } else if (signalType === 'threshold') {
             if (s >= buyThresh) return 'rgba(34, 197, 94, 0.15)'; // Green
             if (s <= sellThresh) return 'rgba(239, 68, 68, 0.15)'; // Red
             return 'rgba(234, 179, 8, 0.15)'; // Yellow
        } else {
             if (s > resMean + resStd * 0.5) return 'rgba(34, 197, 94, 0.15)'; // Green
             if (s < resMean - resStd * 0.5) return 'rgba(239, 68, 68, 0.15)'; // Red
             return 'rgba(234, 179, 8, 0.15)'; // Yellow
        }
      }),
      borderWidth: 0,
      yAxisID: 'y_highlight',
      barPercentage: 1.0,
      categoryPercentage: 1.0,
      animation: false,
    });
  }

  datasets.push({
    label: 'Actual Price',
    data: data.actual_prices,
    borderColor: '#00d4ff', // Cyan
    backgroundColor: '#00d4ff',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.1,
    yAxisID: 'y',
  });

  if (data.model_predictions) {
    data.model_predictions.forEach((modelData, idx) => {
      const color = CHART_COLORS[idx % CHART_COLORS.length];
      datasets.push({
        label: modelData.name,
        data: modelData.predicted_prices,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
        yAxisID: 'y',
      });
    });
  }

  // Add Oscillator Signal Dataset if available
  if (paddedSignals) {
    datasets.push({
      label: 'Oscillator Signal',
      data: paddedSignals,
      borderColor: 'rgba(245, 158, 11, 0.5)', // Warning/Orange with opacity
      backgroundColor: 'rgba(245, 158, 11, 0.5)',
      borderWidth: 1.5,
      borderDash: [5, 5],
      pointRadius: 0,
      tension: 0.1,
      yAxisID: 'y1', // Secondary axis
    });

    if (signalType === 'threshold') {
      datasets.push({
          label: 'Buy Threshold',
          data: new Array(chartDataLength).fill(buyThresh),
          borderColor: 'rgba(34, 197, 94, 0.5)',
          borderWidth: 1,
          borderDash: [2, 2],
          pointRadius: 0,
          yAxisID: 'y1'
      });
      datasets.push({
          label: 'Sell Threshold',
          data: new Array(chartDataLength).fill(sellThresh),
          borderColor: 'rgba(239, 68, 68, 0.5)',
          borderWidth: 1,
          borderDash: [2, 2],
          pointRadius: 0,
          yAxisID: 'y1'
      });
    }
  }

  const chartData = {
    labels: data.dates,
    datasets: datasets,
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#e8edf5',
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
      zoom: {
        pan: {
          enabled: true,
          mode: 'x',
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true
          },
          mode: 'x',
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#8892a8', maxTicksLimit: 15 },
        grid: { color: 'rgba(255, 255, 255, 0.04)' }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        ticks: { color: '#8892a8' },
        grid: { color: 'rgba(255, 255, 255, 0.04)' }
      },
      y1: {
        type: 'linear',
        display: showSignal && data.signals && data.signals.length > 0,
        position: 'right',
        min: -1.2,
        max: 1.2,
        ticks: { color: 'rgba(245, 158, 11, 0.8)' },
        grid: { drawOnChartArea: false }, 
      },
      y_highlight: {
        type: 'linear',
        display: false, // Invisible axis
        position: 'right',
        min: 0,
        max: 1,
      }
    },
  };

  return <Chart type="line" options={options} data={chartData} />;
}
