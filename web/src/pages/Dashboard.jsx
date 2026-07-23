import { useState, useEffect } from 'react';
import axios from 'axios';
import PriceChart from '../components/PriceChart';
import MetricCard from '../components/MetricCard';

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [availablePrimaryModels, setAvailablePrimaryModels] = useState([]);
  const [availableOscillatorModels, setAvailableOscillatorModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedOscillator, setSelectedOscillator] = useState(null);
  
  const [tickerInput, setTickerInput] = useState('BTC-USD');
  const [activeTicker, setActiveTicker] = useState('BTC-USD');
  
  // Fetch available models first
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await axios.get('/api/models');
        
        const primaryModels = response.data.models
          .map(m => m.filename)
          .filter(name => !name.toLowerCase().includes('oscillator'));
          
        const oscillatorModels = response.data.models
          .map(m => m.filename)
          .filter(name => name.toLowerCase().includes('oscillator'));
          
        setAvailablePrimaryModels(primaryModels);
        setAvailableOscillatorModels(oscillatorModels);
        
        // Select active primary model by default
        if (response.data.active?.primary) {
          setSelectedModels([response.data.active.primary]);
        } else if (primaryModels.length > 0) {
          setSelectedModels([primaryModels[0]]);
        }
        
        // Select active oscillator model by default
        if (response.data.active?.oscillator) {
          setSelectedOscillator(response.data.active.oscillator);
        } else if (oscillatorModels.length > 0) {
          setSelectedOscillator(oscillatorModels[0]);
        }
      } catch (err) {
        console.error("Failed to fetch models", err);
      }
    };
    fetchModels();
  }, []);

  // Fetch prediction data when selected models change
  useEffect(() => {
    if (selectedModels.length === 0) return;
    
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams();
        selectedModels.forEach(m => params.append('models', m));
        if (selectedOscillator) {
          params.append('oscillator', selectedOscillator);
        }
        if (activeTicker) {
          params.append('ticker', activeTicker);
        }
        
        const response = await axios.get(`/api/data?${params.toString()}`);
        setData(response.data);
      } catch (err) {
        setError(err.response?.data?.error || err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [selectedModels, selectedOscillator, activeTicker]);

  const togglePrimaryModel = (model) => {
    setSelectedModels(prev => 
      prev.includes(model) 
        ? prev.filter(m => m !== model)
        : [...prev, model]
    );
  };

  // Find the primary model's metrics or fallback to the first available
  const primaryMetrics = data?.model_predictions?.[0]?.metrics;

  return (
    <section className="section active" id="section-dashboard">
      <div className="page-header">
        <h2>Prediction Dashboard</h2>
        <p className="header-subtitle">Real-time price predictions and oscillator signals</p>
      </div>
      
      {error && <p style={{ color: 'var(--danger)', marginBottom: '20px' }}>Error loading data: {error}</p>}
      
      {data && (
        <>
          <div className="grid-4 metrics-row" id="metrics-row">
            <MetricCard label="Ticker" value={data.ticker || '—'} />
            <MetricCard label="Data Points" value={data.test_points || 0} />
            <MetricCard label="MAE" value={primaryMetrics?.MAE ? primaryMetrics.MAE.toFixed(4) : '—'} />
            <MetricCard 
              label="Future Signal" 
              value={data.next_signal !== undefined ? data.next_signal.toFixed(4) : '—'} 
            />
          </div>

          <div className="card" id="chart-card">
            <div className="card-header" style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
              <h3>Price Prediction Overlay</h3>
              <div>
                <strong style={{ display: 'block', marginBottom: '8px', fontSize: '0.85em', color: 'var(--text-muted)' }}>PRIMARY MODELS</strong>
                <div className="model-pills-container" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                  {availablePrimaryModels.map(model => (
                    <button 
                      key={model}
                      className={`btn btn-sm ${selectedModels.includes(model) ? 'btn-primary' : 'btn-secondary'}`}
                      onClick={() => togglePrimaryModel(model)}
                    >
                      {model}
                    </button>
                  ))}
                </div>
              </div>
              
              <div>
                <strong style={{ display: 'block', marginBottom: '8px', fontSize: '0.85em', color: 'var(--text-muted)' }}>TARGET ASSET</strong>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <input 
                    type="text" 
                    className="form-control" 
                    value={tickerInput}
                    onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
                    placeholder="e.g. AAPL, BTC-USD, TSLA"
                    style={{ maxWidth: '200px' }}
                  />
                  <button 
                    className="btn btn-primary btn-sm"
                    onClick={() => setActiveTicker(tickerInput)}
                  >
                    Apply Ticker
                  </button>
                </div>
              </div>
              
              {availableOscillatorModels.length > 0 && (
                <div>
                  <strong style={{ display: 'block', marginBottom: '8px', fontSize: '0.85em', color: 'var(--text-muted)' }}>OSCILLATOR MODEL</strong>
                  <div className="model-pills-container" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                    {availableOscillatorModels.map(model => (
                      <button 
                        key={model}
                        className={`btn btn-sm ${selectedOscillator === model ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => setSelectedOscillator(model === selectedOscillator ? null : model)}
                      >
                        {model}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <div className="chart-container" style={{ position: 'relative', height: '500px', padding: '20px' }}>
              {loading && (
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(10, 14, 26, 0.7)', zIndex: 10 }}>
                  <p>Loading predictions...</p>
                </div>
              )}
              <PriceChart data={data} />
            </div>
          </div>
        </>
      )}
    </section>
  );
}
