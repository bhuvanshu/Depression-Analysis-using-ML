import { useState, useEffect, useMemo } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, BarElement, Title, Tooltip, Legend, Filler, ArcElement
} from 'chart.js';
import { Bar, Line, Pie } from 'react-chartjs-2';
import { TrendingUp, Brain, Info, BarChart3, Activity } from 'lucide-react';
import Card from '../../components/common/Card';
import { getDashboardStudents } from '../../services/api';
import { getChartOptions } from '../../config/chartDefaults';
import { useTheme } from '../../context/ThemeContext';
import './ReportsPage.css';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler, ArcElement
);


function IntelligenceSummary({ students }) {
  const summary = useMemo(() => {
    if (!students.length) return { stressor: 'N/A', stressorAvg: '0', topDept: 'N/A', participation: '0' };

    const count = students.length;
    const sums = { 'Academic Pressure': 0, 'Financial Stress': 0 };
    const deptRisk = {};

    students.forEach(s => {
      sums['Academic Pressure'] += (s.academicPressure || 0);
      sums['Financial Stress'] += (s.financialStress || 0);

      const d = s.department || 'Unknown';
      if (!deptRisk[d]) deptRisk[d] = 0;
      if (s.riskLevel === 'High') deptRisk[d]++;
    });

    // Primary stressor
    const stressorEntries = Object.entries(sums).map(([k, v]) => [k, v / count]);
    stressorEntries.sort((a, b) => b[1] - a[1]);
    const [stressor, stressorAvg] = stressorEntries[0];

    // Top risk department
    const deptEntries = Object.entries(deptRisk).filter(([, v]) => v > 0);
    deptEntries.sort((a, b) => b[1] - a[1]);
    const topDept = deptEntries.length > 0 ? deptEntries[0][0] : 'None';

    return { stressor, stressorAvg: stressorAvg.toFixed(1), topDept, participation: count };
  }, [students]);

  return (
    <Card elevated className="intelligence-summary">
      <div className="summary-header">
        <Info size={20} />
        <h3>System Intelligence Summary</h3>
      </div>
      <div className="summary-grid">
        <div className="summary-item">
          <label>Primary Stressor</label>
          <span>{summary.stressor} (Avg {summary.stressorAvg})</span>
        </div>
        <div className="summary-item">
          <label>Top Risk Department</label>
          <span style={{ color: summary.topDept !== 'None' ? 'var(--accent-danger)' : 'inherit' }}>
            {summary.topDept}
          </span>
        </div>
        <div className="summary-item">
          <label>Students Screened</label>
          <span>{summary.participation} students</span>
        </div>
      </div>
    </Card>
  );
}

export default function ReportsPage() {
  const { theme } = useTheme();
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
        data: [(sums.academic / count).toFixed(2), (sums.financial / count).toFixed(2), (sums.satisfaction / count).toFixed(2)],
        backgroundColor: ['rgba(99, 102, 241, 0.7)', 'rgba(239, 68, 68, 0.7)', 'rgba(34, 197, 94, 0.7)'],
        borderRadius: 8,
      }]
    };
  }, [students]);
  // ── C. Department-wise Risk Bar ──
  const deptChartData = useMemo(() => {
    const deptMap = {};
    students.forEach(s => {
      const d = s.department || 'Unknown';
      if (!deptMap[d]) deptMap[d] = { high: 0, moderate: 0, low: 0 };
      
      if (s.riskLevel === 'High') deptMap[d].high++;
      else if (s.riskLevel === 'Moderate') deptMap[d].moderate++;
      else deptMap[d].low++;
    });

    const labels = Object.keys(deptMap);
    return {
      labels: labels.map(d => d.length > 12 ? d.slice(0, 12) + '…' : d),
      datasets: [
        {
          label: 'High Priority',
          data: labels.map(d => deptMap[d].high),
          backgroundColor: 'rgba(239, 68, 68, 0.7)',
          borderRadius: 4
        },
        {
          label: 'Moderate Priority',
          data: labels.map(d => deptMap[d].moderate),
          backgroundColor: 'rgba(245, 158, 11, 0.7)',
          borderRadius: 4
        },
        {
          label: 'Low Priority',
          data: labels.map(d => deptMap[d].low),
          backgroundColor: 'rgba(34, 197, 94, 0.7)',
          borderRadius: 4
        }
      ]
    };
  }, [students]);

  // ── D. Study Satisfaction Distribution (Pie Chart) ──
  const satisfactionPieData = useMemo(() => {
    const counts = Array(6).fill(0);
    students.forEach(s => {
      const val = s.studySatisfaction;
      if (val !== undefined && val !== null && val >= 0 && val <= 5) {
        counts[val]++;
      }
    });

    return {
      labels: ['Very Low (0)', 'Low (1)', 'Below Avg (2)', 'Average (3)', 'Good (4)', 'Excellent (5)'],
      datasets: [{
        data: counts,
        backgroundColor: [
          'rgba(239, 68, 68, 0.75)',   // Very Low
          'rgba(249, 115, 22, 0.75)',  // Low
          'rgba(245, 158, 11, 0.75)',  // Below Avg
          'rgba(99, 102, 241, 0.75)',  // Average
          'rgba(59, 130, 246, 0.75)',  // Good
          'rgba(34, 197, 94, 0.75)'    // Excellent
        ],
        borderColor: [
          '#EF4444', '#F97316', '#F59E0B', '#6366F1', '#3B82F6', '#22C55E'
        ],
        borderWidth: 2
      }]
    };
  }, [students]);

  // ── E. Risk Distribution (Bar Chart) ──
  const riskBarData = useMemo(() => {
    let low = 0, moderate = 0, high = 0;
    students.forEach(s => {
      if (s.riskLevel === 'High') high++;
      else if (s.riskLevel === 'Moderate') moderate++;
      else if (s.riskLevel === 'Low') low++;
    });

    return {
      labels: ['Low Priority', 'Moderate Priority', 'High Priority'],
      datasets: [{
        label: 'Student Count',
        data: [low, moderate, high],
        backgroundColor: [
          'rgba(34, 197, 94, 0.7)',
          'rgba(245, 158, 11, 0.7)',
          'rgba(239, 68, 68, 0.7)'
        ],
        borderColor: ['#22C55E', '#F59E0B', '#EF4444'],
        borderWidth: 2,
        borderRadius: 6
      }]
    };
  }, [students]);

  // ── F. Suicidal Thoughts Flag Rate (Pie Chart) ──
  const suicidalThoughtsPieData = useMemo(() => {
    let yesCount = 0;
    let noCount = 0;

    students.forEach(s => {
      if (s.suicidalThoughts === true || s.suicidalThoughts === 'Yes' || s.suicidalThoughts === 1) {
        yesCount++;
      } else {
        noCount++;
      }
    });

    return {
      labels: ['Reported Thoughts (Yes)', 'No Reported Thoughts (No)'],
      datasets: [{
        data: [yesCount, noCount],
        backgroundColor: [
          'rgba(239, 68, 68, 0.75)',  // Red
          'rgba(34, 197, 94, 0.75)'   // Green
        ],
        borderColor: [
          '#EF4444',
          '#22C55E'
        ],
        borderWidth: 2
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
                ...getChartOptions(theme, 'linear'),
                plugins: {
                  ...getChartOptions(theme, 'linear').plugins,
                  legend: { display: false }
                }
              }}
            />
          </div>
        </Card>

        <Card elevated className="report-card">
          <div className="report-card-header">
            <Activity size={18} style={{ color: 'var(--accent-warning)' }} />
            <h3>Study Satisfaction Distribution</h3>
          </div>
          <div className="report-chart-container">
            <Pie
              data={satisfactionPieData}
              options={{
                ...getChartOptions(theme, 'radial'),
                plugins: {
                  ...getChartOptions(theme, 'radial').plugins,
                  legend: {
                    position: 'bottom',
                    onClick: () => {},
                    labels: {
                      ...getChartOptions(theme, 'radial').plugins.legend.labels,
                      font: { family: 'Inter', size: 11 }
                    }
                  }
                }
              }}
            />
          </div>
        </Card>

        <Card elevated className="report-card">
          <div className="report-card-header">
            <TrendingUp size={18} style={{ color: 'var(--accent-danger)' }} />
            <h3>Priority Distribution Breakdown</h3>
          </div>
          <div className="report-chart-container">
            <Bar
              data={riskBarData}
              options={{
                ...getChartOptions(theme, 'linear'),
                scales: {
                  x: {
                    ...getChartOptions(theme, 'linear').scales.x,
                    grid: { display: false },
                    ticks: { ...getChartOptions(theme, 'linear').scales.x.ticks, font: { size: 11 } }
                  },
                  y: {
                    ...getChartOptions(theme, 'linear').scales.y,
                    ticks: { ...getChartOptions(theme, 'linear').scales.y.ticks, stepSize: 1 }
                  }
                },
                plugins: {
                  ...getChartOptions(theme, 'linear').plugins,
                  legend: { display: false }
                }
              }}
            />
          </div>
        </Card>

        <Card elevated className="report-card">
          <div className="report-card-header">
            <Brain size={18} style={{ color: 'var(--accent-danger)' }} />
            <h3>Suicidal Thoughts Flag Rate</h3>
          </div>
          <div className="report-chart-container">
            <Pie
              data={suicidalThoughtsPieData}
              options={{
                ...getChartOptions(theme, 'radial'),
                plugins: {
                  ...getChartOptions(theme, 'radial').plugins,
                  legend: {
                    position: 'bottom',
                    onClick: () => {},
                    labels: {
                      ...getChartOptions(theme, 'radial').plugins.legend.labels,
                      font: { family: 'Inter', size: 11 }
                    }
                  }
                }
              }}
            />
          </div>
        </Card>

        <Card elevated className="report-card report-card-full">
          <div className="report-card-header">
            <BarChart3 size={18} style={{ color: 'var(--accent-primary)' }} />
            <h3>Department-wise Priority Breakdown</h3>
          </div>
          <div className="report-chart-container">
            <Bar
              data={deptChartData}
              options={{
                ...getChartOptions(theme, 'linear'),
                scales: {
                  x: {
                    ...getChartOptions(theme, 'linear').scales.x,
                    stacked: true,
                    grid: { display: false },
                    ticks: { ...getChartOptions(theme, 'linear').scales.x.ticks, font: { size: 11 } }
                  },
                  y: {
                    ...getChartOptions(theme, 'linear').scales.y,
                    stacked: true,
                    ticks: { ...getChartOptions(theme, 'linear').scales.y.ticks, stepSize: 1 }
                  }
                }
              }}
            />
          </div>
        </Card>
      </div>

      <IntelligenceSummary students={students} />
    </div>
  );
}
