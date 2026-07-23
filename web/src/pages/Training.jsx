import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

export default function Training() {
  const [config, setConfig] = useState(null);
  const [status, setStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [modelName, setModelName] = useState("");
  const terminalEndRef = useRef(null);

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [status?.logs, status?.progress_log]);

  useEffect(() => {
    // Fetch initial config
    axios.get('/api/config').then(res => {
      setConfig(res.data);
      const dateStr = new Date().toISOString().replace(/T/, '_').replace(/:/g, '').split('.')[0];
      setModelName(`${res.data.MODEL_TYPE}_${dateStr}`);
    }).catch(console.error);
  }, []);

  useEffect(() => {
    let interval;
    if (isTraining) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get('/api/training_status');
          setStatus(res.data);
          if (!res.data.is_training) {
            setIsTraining(false);
          }
        } catch (err) {
          console.error(err);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isTraining]);

  const startTraining = async () => {
    try {
      const oscEpochs = config.OSCILLATOR_EPOCHS || 10;
      await axios.post(`/api/train?model_type=${config.MODEL_TYPE}&epochs=${config.EPOCHS}&oscillator_epochs=${oscEpochs}&model_name=${modelName}`, config);
      setIsTraining(true);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <section className="section active">
      <div className="page-header">
        <h2>Model Training</h2>
        <p className="header-subtitle">Configure hyperparameters and train a new dual-stream model</p>
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-header">
          <h3>Training Configuration</h3>
        </div>
        <div style={{ padding: '20px' }}>
          {config ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
              <div>
                <label>Ticker Primary: </label>
                <input 
                  type="text" 
                  value={config.TICKER_PRIMARY} 
                  onChange={e => setConfig({...config, TICKER_PRIMARY: e.target.value})} 
                  style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-subtle)', color: 'white', padding: '5px', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label>Primary Epochs: </label>
                <input 
                  type="number" 
                  value={config.EPOCHS} 
                  onChange={e => setConfig({...config, EPOCHS: parseInt(e.target.value) || 0})} 
                  style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-subtle)', color: 'white', padding: '5px', borderRadius: '4px', marginRight: '15px' }}
                />
                <label>Oscillator Epochs: </label>
                <input 
                  type="number" 
                  value={config.OSCILLATOR_EPOCHS || 10} 
                  onChange={e => setConfig({...config, OSCILLATOR_EPOCHS: parseInt(e.target.value) || 0})} 
                  style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-subtle)', color: 'white', padding: '5px', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label>Model Architecture: </label>
                <select 
                  value={config.MODEL_TYPE} 
                  onChange={e => {
                    const newType = e.target.value;
                    setConfig({...config, MODEL_TYPE: newType});
                    const dateStr = new Date().toISOString().replace(/T/, '_').replace(/:/g, '').split('.')[0];
                    setModelName(`${newType}_${dateStr}`);
                  }} 
                  style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-subtle)', color: 'white', padding: '5px', borderRadius: '4px', width: '100%', marginTop: '5px' }}
                >
                  <option value="lstm">Single-Stream LSTM</option>
                  <option value="mtl">Multi-Task Learning (MTL)</option>
                  <option value="extended_mtl">Extended MTL (Momentum-Rotation)</option>
                </select>
              </div>
              <div>
                <label>Model Name: </label>
                <input 
                  type="text" 
                  value={modelName} 
                  onChange={e => setModelName(e.target.value)} 
                  style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-subtle)', color: 'white', padding: '5px', borderRadius: '4px', width: '100%', marginTop: '5px' }}
                />
              </div>
              <button 
                className="btn btn-primary" 
                onClick={startTraining} 
                disabled={isTraining}
                style={{ alignSelf: 'flex-start', padding: '10px 20px', borderRadius: '8px', cursor: 'pointer' }}
              >
                {isTraining ? 'Training in progress...' : 'Start Training'}
              </button>
            </div>
          ) : (
            <p>Loading config...</p>
          )}
        </div>
      </div>

      {status && (status.is_training || (status.logs && status.logs.length > 0)) && (
        <div className="card terminal-card" style={{ marginTop: '20px', background: '#0d1117', border: '1px solid #30363d', borderRadius: '10px', overflow: 'hidden', boxShadow: '0 10px 30px rgba(0,0,0,0.5)' }}>
          <div className="terminal-header" style={{ display: 'flex', alignItems: 'center', padding: '10px 15px', background: '#161b22', borderBottom: '1px solid #30363d' }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#ff5f56' }}></div>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#ffbd2e' }}></div>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#27c93f' }}></div>
            </div>
            <div style={{ margin: '0 auto', color: '#8b949e', fontSize: '12px', fontFamily: 'monospace' }}>
              bash - training_pipeline
            </div>
            {/* placeholder to balance flex */}
            <div style={{ width: '52px' }}></div>
          </div>
          <div 
            className="terminal-body" 
            style={{ 
              padding: '20px', 
              height: '400px', 
              overflowY: 'auto', 
              fontFamily: 'Consolas, Monaco, "Courier New", monospace',
              fontSize: '13px',
              color: '#c9d1d9',
              lineHeight: '1.6'
            }}
          >
            {status.logs && status.logs.map((log, i) => (
              <div key={i} style={{ 
                color: log.includes('[ERROR]') ? '#ff7b72' : 
                       log.includes('[SYSTEM]') ? '#79c0ff' : 
                       log.includes('[PRIMARY]') ? '#a5d6ff' : 
                       log.includes('[OSCILLATOR]') ? '#d2a8ff' : '#c9d1d9',
                marginBottom: '4px'
              }}>
                <span style={{ color: '#8b949e', marginRight: '12px' }}>$</span>
                {log}
              </div>
            ))}
            
            {status.progress_log && (
              <div style={{ color: '#3fb950', marginTop: '12px', marginBottom: '8px' }}>
                <span style={{ color: '#8b949e', marginRight: '12px' }}>&gt;</span>
                {status.progress_log}
              </div>
            )}
            
            {/* Mini stats display during epochs */}
            {status.is_training && status.epoch > 0 && (
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(4, 1fr)', 
                gap: '10px', 
                padding: '10px', 
                background: 'rgba(139, 148, 158, 0.1)', 
                borderRadius: '6px',
                marginTop: '10px',
                marginBottom: '10px'
              }}>
                <div><span style={{color: '#8b949e'}}>Epoch:</span> {status.epoch}/{status.total_epochs}</div>
                <div><span style={{color: '#8b949e'}}>Loss:</span> {status.loss?.toFixed(4)}</div>
                <div><span style={{color: '#8b949e'}}>Val Loss:</span> {status.val_loss?.toFixed(4)}</div>
                <div><span style={{color: '#8b949e'}}>MAE:</span> {status.mae?.toFixed(4)}</div>
              </div>
            )}

            {status.is_training && (
              <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ color: '#8b949e' }}>$</span>
                <span style={{ display: 'inline-block', width: '8px', height: '16px', background: '#c9d1d9', animation: 'blink 1s step-end infinite' }}></span>
              </div>
            )}
            <div ref={terminalEndRef} />
          </div>
        </div>
      )}

      <style>{`
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        .terminal-body::-webkit-scrollbar {
          width: 8px;
        }
        .terminal-body::-webkit-scrollbar-track {
          background: #0d1117;
        }
        .terminal-body::-webkit-scrollbar-thumb {
          background: #30363d;
          border-radius: 4px;
        }
      `}</style>
    </section>
  );
}
