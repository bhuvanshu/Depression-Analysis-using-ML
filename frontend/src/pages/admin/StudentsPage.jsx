import { useState, useMemo } from 'react';
import { Search, Filter, ChevronDown, Download, User, MoreVertical } from 'lucide-react';
import Card from '../../components/common/Card';
import Button from '../../components/common/Button';
import RiskBadge from '../../components/common/RiskBadge';
import { MOCK_STUDENTS, MOCK_RESULTS, MOCK_RESPONSES } from '../../data/mockData';
import './StudentsPage.css';

export default function StudentsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [riskFilter, setRiskFilter] = useState('All');

  // Merge student + result data for table
  const tableData = useMemo(() => {
    return MOCK_STUDENTS.map(student => {
      const response = MOCK_RESPONSES.find(r => r.student_id === student.student_id);
      const result = response ? MOCK_RESULTS.find(r => r.response_id === response.response_id) : null;
      return { ...student, response, result };
    }).filter(s => s.result);
  }, []);

  const filteredData = useMemo(() => {
    return tableData.filter(s => {
      const matchesSearch = !searchQuery.trim() || 
        s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.enrollment_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.department.toLowerCase().includes(searchQuery.toLowerCase());
        
      const matchesRisk = riskFilter === 'All' || s.result?.risk_level === riskFilter;
      
      return matchesSearch && matchesRisk;
    });
  }, [searchQuery, riskFilter, tableData]);

  const getInitials = (name) => name ? name.split(' ').map(n => n[0]).join('').toUpperCase() : '?';
  const getScoreClass = (level) => level?.toLowerCase() || 'low';

  return (
    <div className="students-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Student Records</h1>
          <p className="page-subtitle">Manage and monitor student screenings</p>
        </div>
        <div className="page-actions">
          <Button variant="secondary" icon={Download}>Export List</Button>
        </div>
      </div>

      <Card elevated className="students-table-container">
        {/* Table Toolbar */}
        <div className="table-toolbar">
          <div className="toolbar-search">
            <div className="input-wrapper">
              <Search size={18} className="input-icon" />
              <input
                type="text"
                className="input-field has-icon"
                placeholder="Search name, ID, or department..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>
          <div className="toolbar-filters">
            <div className="filter-group">
              <span className="filter-label">Risk Level:</span>
              <select 
                className="input-field" 
                value={riskFilter}
                onChange={(e) => setRiskFilter(e.target.value)}
                style={{ width: '140px' }}
              >
                <option value="All">All Levels</option>
                <option value="High">High Risk</option>
                <option value="Moderate">Moderate</option>
                <option value="Low">Low Risk</option>
              </select>
            </div>
          </div>
        </div>

        {/* Data Table */}
        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Student</th>
                <th>Enrollment ID</th>
                <th>Department</th>
                <th>Risk Level</th>
                <th>Score</th>
                <th>Date Screened</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.map((row) => (
                <tr key={row.student_id}>
                  <td>
                    <div className="table-student-name">
                      <div className="table-avatar">{getInitials(row.name)}</div>
                      <div>
                        <span className="table-name-text">{row.name}</span>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{row.degree}</div>
                      </div>
                    </div>
                  </td>
                  <td><span className="table-enrollment">{row.enrollment_id}</span></td>
                  <td>{row.department}</td>
                  <td><RiskBadge level={row.result?.risk_level} /></td>
                  <td>
                    <span className={`table-score score-${getScoreClass(row.result?.risk_level)}`}>
                      {((row.result?.probability_score || 0) * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td>{row.response?.submitted_at ? new Date(row.response.submitted_at).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }) : '—'}</td>
                  <td>
                    <Button variant="ghost" size="sm" icon={User}>View</Button>
                  </td>
                </tr>
              ))}
              {filteredData.length === 0 && (
                <tr>
                  <td colSpan="7" className="table-empty-state">
                    No students found matching your criteria
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="table-pagination">
          <span className="table-pagination-info">
            Showing {filteredData.length} of {tableData.length} entries
          </span>
          <div className="table-pagination-controls">
            <Button variant="ghost" size="sm" disabled>Previous</Button>
            <Button variant="ghost" size="sm" disabled>Next</Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
