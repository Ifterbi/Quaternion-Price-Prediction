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

  let paddedSignals = null;
  if (showSignal && data.signals && data.signals.length > 0) {
    paddedSignals = new Array(data.actual_prices.length - data.signals.length)
      .fill(null)
      .concat(data.signals);
  }

  const datasets = [];

  // Add Vertical Highlight Bar Dataset (this goes first to act as a background)
  if (paddedSignals) {
    datasets.push({
      type: 'bar',
      label: 'Signal Highlight',
      data: paddedSignals.map(s => (s !== null && s !== undefined) ? 1 : 0),
      backgroundColor: paddedSignals.map(s => {
        if (s === null || s === undefined) return 'transparent';
        if (s === 0) return 'rgba(234, 179, 8, 0.15)'; // Yellow
        return s < 0 ? 'rgba(239, 68, 68, 0.15)' : 'rgba(34, 197, 94, 0.15)'; // Red for sell, Green for buy
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
