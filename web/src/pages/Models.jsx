import { useState, useEffect } from 'react';
import axios from 'axios';

export default function Models() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get('/api/models');
      setModels(response.data.models);
    } catch (err) {
      console.error(err);
      setMessage({ type: 'danger', text: 'Failed to fetch models: ' + err.message });
    } finally {
      setLoading(false);
    }
  };

  const activateModel = async (filename, type) => {
    try {
      await axios.post('/api/activate_model', null, { params: { filename, model_type: type } });
      setMessage({ type: 'success', text: `Activated ${type} model: ${filename}` });
      fetchModels();
    } catch (err) {
      console.error(err);
      setMessage({ type: 'danger', text: 'Failed to activate model: ' + (err.response?.data?.detail || err.message) });
    }
  };

  const uploadModel = async (file, type) => {
    if (!file) return;
    setUploading(true);
    setMessage({ type: 'info', text: `Uploading ${file.name}...` });
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      await axios.post(`/api/upload_model?model_type=${type}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setMessage({ type: 'success', text: `Model uploaded: ${file.name}` });
      fetchModels();
    } catch (err) {
      console.error(err);
      setMessage({ type: 'danger', text: 'Upload failed: ' + (err.response?.data?.detail || err.message) });
    } finally {
      setUploading(false);
    }
  };

  return (
    <section className="section active">
      <div className="page-header">
        <h2>Model Management</h2>
        <p className="header-subtitle">View, activate, and upload trained models</p>
      </div>

      {message && (
        <div style={{ padding: '10px', marginBottom: '20px', borderRadius: '4px', backgroundColor: message.type === 'danger' ? 'rgba(239, 68, 68, 0.2)' : message.type === 'success' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(59, 130, 246, 0.2)', color: message.type === 'danger' ? 'var(--danger)' : message.type === 'success' ? 'var(--success)' : 'var(--accent-blue)', border: `1px solid ${message.type === 'danger' ? 'var(--danger)' : message.type === 'success' ? 'var(--success)' : 'var(--accent-blue)'}` }}>
          {message.text}
        </div>
      )}

      <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
        <div className="card">
          <div className="card-header">
            <h3>Upload Primary Model</h3>
          </div>
          <div style={{ padding: '20px', textAlign: 'center' }}>
            <p className="text-muted" style={{ marginBottom: '15px' }}>LSTM/Dual-Stream (.keras)</p>
            <input 
              type="file" 
              accept=".keras,.h5" 
              id="upload-primary"
              style={{ display: 'none' }}
              onChange={(e) => uploadModel(e.target.files[0], 'primary')}
            />
            <label htmlFor="upload-primary" className="btn btn-primary" style={{ cursor: 'pointer', display: 'inline-block' }}>
              Select File
            </label>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Upload Oscillator Model</h3>
          </div>
          <div style={{ padding: '20px', textAlign: 'center' }}>
            <p className="text-muted" style={{ marginBottom: '15px' }}>Residual/Signal (.keras)</p>
            <input 
              type="file" 
              accept=".keras,.h5" 
              id="upload-oscillator"
              style={{ display: 'none' }}
              onChange={(e) => uploadModel(e.target.files[0], 'oscillator')}
            />
            <label htmlFor="upload-oscillator" className="btn btn-secondary" style={{ cursor: 'pointer', display: 'inline-block' }}>
              Select File
            </label>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Saved Models</h3>
        </div>
        <div className="model-list">
          {loading ? (
            <p style={{ padding: '20px' }}>Loading models...</p>
          ) : models.length === 0 ? (
            <p style={{ padding: '20px' }} className="text-muted">No models found.</p>
          ) : (
            models.map((model) => (
              <div key={model.filename} className="model-list-item" style={{ display: 'flex', justifyContent: 'space-between', padding: '15px', borderBottom: '1px solid var(--border-subtle)', alignItems: 'center' }}>
                <div>
                  <strong style={{ fontSize: '1.1em', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {model.filename}
                    {model.is_active_primary && <span style={{ fontSize: '0.7em', padding: '2px 6px', borderRadius: '10px', backgroundColor: 'rgba(34, 197, 94, 0.2)', color: 'var(--success)' }}>Active Primary</span>}
                    {model.is_active_oscillator && <span style={{ fontSize: '0.7em', padding: '2px 6px', borderRadius: '10px', backgroundColor: 'rgba(245, 158, 11, 0.2)', color: 'var(--warning)' }}>Active Oscillator</span>}
                  </strong>
                  <div style={{ fontSize: '0.85em', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    Size: {model.size_human} | Modified: {new Date(model.modified).toLocaleString()}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button 
                    className="btn btn-sm btn-primary" 
                    onClick={() => activateModel(model.filename, 'primary')}
                    disabled={model.is_active_primary}
                  >
                    {model.is_active_primary ? 'Active Primary' : 'Set Primary'}
                  </button>
                  <button 
                    className="btn btn-sm btn-secondary" 
                    onClick={() => activateModel(model.filename, 'oscillator')}
                    disabled={model.is_active_oscillator}
                  >
                    {model.is_active_oscillator ? 'Active Oscillator' : 'Set Oscillator'}
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
}
