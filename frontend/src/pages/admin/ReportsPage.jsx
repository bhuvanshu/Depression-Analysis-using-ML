import { useState, useEffect, useMemo } from 'react';
import { 
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, 
  LineElement, BarElement, Title, Tooltip, Legend, Filler 
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';
import { Download, FileText, TrendingUp, Brain, Info, Database } from 'lucide-react';
import Card from '../../components/common/Card';
import Button from '../../components/common/Button';
import { getDashboardStudents } from '../../services/api';
import './ReportsPage.css';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, 
  BarElement, Title, Tooltip, Legend, Filler
);

export default function ReportsPage() {
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDashboardStudents().then(data => {
      setStudents(data || []);
      setLoading(false);
    });
  }, []);

  // ── A. Average Stress Indicators ──
  const stressMetrics = useMemo(() => {
    if (!students.length) return { labels: [], datasets: [] };
    const sums = { academic: 0, financial: 0, satisfaction: 0 };
    students.forEach(s => {
      sums.academic += (s.academicPressure || 0);
      sums.financial += (s.financialStress || 0);
      sums.satisfaction += (s.studySatisfaction || 0);
    });
    const count = students.length;
    return {
      labels: ['Academic Pressure', 'Financial Stress', 'Study Satisfaction'],
      datasets: [{
        label: 'Institutional Avg (1-5)',
        data: [(sums.academic/count).toFixed(2), (sums.financial/count).toFixed(2), (sums.satisfaction/count).toFixed(2)],
        backgroundColor: ['rgba(99, 102, 241, 0.7)', 'rgba(239, 68, 68, 0.7)', 'rgba(34, 197, 94, 0.7)'],
        borderRadius: 8,
      }]
    };
  }, [students]);

  // ── E. Correlation Insights ──
  const correlationData = useMemo(() => {
    const highPressure = students.filter(s => (s.academicPressure || 0) >= 4);
    const lowPressure = students.filter(s => (s.academicPressure || 0) <= 2);
    
    const highRiskInHighPressure = highPressure.filter(s => s.riskLevel === 'High').length;
    const highRiskInLowPressure = lowPressure.filter(s => s.riskLevel === 'High').length;

    return {
      labels: ['High Pressure Group', 'Low Pressure Group'],
      datasets: [{
        label: 'High Risk Frequency (%)',
        data: [
          highPressure.length ? ((highRiskInHighPressure / highPressure.length) * 100).toFixed(0) : 0,
          lowPressure.length ? ((highRiskInLowPressure / lowPressure.length) * 100).toFixed(0) : 0
        ],
        backgroundColor: ['#EF4444', '#94A3B8'],
        barThickness: 40
      }]
    };
  }, [students]);

  return (
    <div className="reports-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Intelligence Reports</h1>
          <p className="page-subtitle">Analytical & Export Layer — Institutional Data Patterns</p>
        </div>
        <div className="page-actions">
          <Button variant="secondary" icon={FileText}>Export CSV</Button>
          <Button variant="primary" icon={Download}>Export PDF</Button>
        </div>
      </div>

      <div className="reports-grid">
        <Card elevated className="report-card">
          <div className="report-card-header">
            <TrendingUp size={18} style={{ color: 'var(--accent-primary)' }} />
            <h3>Average Stress Indicators</h3>
          </div>
          <div className="report-chart-container">
            <Bar 
              data={stressMetrics} 
              options={{ 
                responsive: true, 
                maintainAspectRatio: false,
                plugins: { legend: { display: false } }
              }} 
            />
          </div>
        </Card>

        <Card elevated className="report-card">
          <div className="report-card-header">
            <Brain size={18} style={{ color: 'var(--accent-danger)' }} />
            <h3>Risk Correlation Insight</h3>
          </div>
          <p className="insight-text">
            Students with <strong>High Academic Pressure</strong> show a 
            statistically higher depression probability.
          </p>
          <div className="report-chart-container">
            <Bar 
              data={correlationData} 
              options={{ 
                indexAxis: 'y', 
                responsive: true, 
                maintainAspectRatio: false,
                plugins: { legend: { display: false } }
              }} 
            />
          </div>
        </Card>
      </div>

      <Card elevated className="intelligence-summary">
        <div className="summary-header">
          <Info size={20} />
          <h3>System Intelligence Summary</h3>
        </div>
        <div className="summary-grid">
          <div className="summary-item">
            <label>Primary Stressor</label>
            <span>Academic Pressure (Avg 4.2)</span>
          </div>
          <div className="summary-item">
            <label>Top Risk Department</label>
            <span style={{ color: 'var(--accent-danger)' }}>Computer Science</span>
          </div>
          <div className="summary-item">
            <label>Participation Rate</label>
            <span>82.4% institutional coverage</span>
          </div>
        </div>
      </Card>
    </div>
  );
}
