import { NavLink } from 'react-router-dom';
import { BarChart2, Brain, Zap, Activity } from 'lucide-react';

export default function Sidebar() {
  return (
    <aside className="sidebar" id="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">Q</div>
        <div className="brand-text">
          <h1>Quaternion AI</h1>
          <p>Price Predictor</p>
        </div>
      </div>

      <nav className="sidebar-nav">
        <NavLink 
          to="/dashboard" 
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <BarChart2 className="nav-icon" size={20} />
          <span>Price Predictor</span>
        </NavLink>
        <NavLink 
          to="/oscillator" 
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <Activity className="nav-icon" size={20} />
          <span>Oscillators</span>
        </NavLink>
        <NavLink 
          to="/models" 
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <Brain className="nav-icon" size={20} />
          <span>Models</span>
        </NavLink>
        <NavLink 
          to="/training" 
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <Zap className="nav-icon" size={20} />
          <span>Training</span>
        </NavLink>
      </nav>

      <div className="sidebar-footer">
        <div className="status-row">
          <span className="status-dot" id="status-dot" style={{ background: 'var(--success)', boxShadow: '0 0 8px var(--success)' }}></span>
          <span id="status-text">Ready</span>
        </div>
        <div className="version-text">Dual-Stream LSTM · FiLM Fusion</div>
      </div>
    </aside>
  );
}
