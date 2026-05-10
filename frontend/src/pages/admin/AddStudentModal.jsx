import React, { useState, useRef } from 'react';
import { UserPlus, Upload, Download, Check, AlertCircle, FileText } from 'lucide-react';
import Modal from '../../components/common/Modal';
import Button from '../../components/common/Button';
import Input from '../../components/common/Input';
import { addStudent, bulkAddStudents } from '../../services/api';
import './AddStudentModal.css';

export default function AddStudentModal({ isOpen, onClose, onSuccess }) {
  const [activeTab, setActiveTab] = useState('single');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  
  // Single Entry State
  const [formData, setFormData] = useState({
    enrollmentId: '',
    name: '',
    age: '',
    gender: 'Male',
    department: '',
    degreeGroup: 'Undergraduate'
  });

  // Bulk Upload State
  const [csvPreview, setCsvPreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError('');
  };

  const handleSingleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      await addStudent({
        ...formData,
        age: parseInt(formData.age)
      });
      setSuccess(true);
      setTimeout(() => {
        onSuccess();
        handleClose();
      }, 1500);
    } catch (err) {
      setError(err.message || 'Failed to add student');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const lines = text.split('\n');
      const headers = lines[0].split(',').map(h => h.trim());
      
      const data = lines.slice(1).filter(line => line.trim()).map(line => {
        const values = line.split(',').map(v => v.trim());
        const student = {};
        headers.forEach((header, index) => {
          student[header] = values[index];
        });
        return student;
      });
      
      setCsvPreview(data);
    };
    reader.readAsText(file);
  };

  const handleBulkSubmit = async () => {
    if (!csvPreview) return;
    setLoading(true);
    setError('');
    
    try {
      await bulkAddStudents(csvPreview);
      setSuccess(true);
      setTimeout(() => {
        onSuccess();
        handleClose();
      }, 1500);
    } catch (err) {
      setError(err.message || 'Failed to import students');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setFormData({
      enrollmentId: '',
      name: '',
      age: '',
      gender: 'Male',
      department: '',
      degreeGroup: 'Undergraduate'
    });
    setCsvPreview(null);
    setSuccess(false);
    setError('');
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Add New Students" maxWidth="600px">
      <div className="add-student-modal">
        {success ? (
          <div className="success-state animate-scale-up">
            <div className="success-icon">
              <Check size={40} />
            </div>
            <h3>Successfully Added!</h3>
            <p>The student records have been updated in the system.</p>
          </div>
        ) : (
          <>
            <div className="modal-tabs">
              <button 
                className={`tab-btn ${activeTab === 'single' ? 'active' : ''}`}
                onClick={() => setActiveTab('single')}
              >
                <UserPlus size={18} />
                Single Entry
              </button>
              <button 
                className={`tab-btn ${activeTab === 'bulk' ? 'active' : ''}`}
                onClick={() => setActiveTab('bulk')}
              >
                <Upload size={18} />
                Bulk Import (CSV)
              </button>
            </div>

            {activeTab === 'single' ? (
              <form className="single-entry-form" onSubmit={handleSingleSubmit}>
                <div className="form-grid">
                  <Input
                    label="Full Name"
                    name="name"
                    placeholder="Enter full name"
                    value={formData.name}
                    onChange={handleInputChange}
                    required
                  />
                  <Input
                    label="Enrollment ID"
                    name="enrollmentId"
                    placeholder="e.g., BT21CSE001"
                    value={formData.enrollmentId}
                    onChange={handleInputChange}
                    required
                  />
                  <Input
                    label="Age"
                    name="age"
                    type="number"
                    placeholder="Enter age"
                    value={formData.age}
                    onChange={handleInputChange}
                    required
                  />
                  <div className="form-group">
                    <label className="input-label">Gender</label>
                    <select 
                      name="gender" 
                      className="input-field"
                      value={formData.gender}
                      onChange={handleInputChange}
                    >
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                  <Input
                    label="Department"
                    name="department"
                    placeholder="e.g., Computer Science"
                    value={formData.department}
                    onChange={handleInputChange}
                    required
                  />
                  <div className="form-group">
                    <label className="input-label">Degree Group</label>
                    <select 
                      name="degreeGroup" 
                      className="input-field"
                      value={formData.degreeGroup}
                      onChange={handleInputChange}
                    >
                      <option value="Undergraduate">Undergraduate</option>
                      <option value="Postgraduate">Postgraduate</option>
                      <option value="Doctorate">Doctorate</option>
                    </select>
                  </div>
                </div>

                {error && <div className="error-message"><AlertCircle size={16} />{error}</div>}

                <div className="modal-footer">
                  <Button variant="ghost" onClick={handleClose}>Cancel</Button>
                  <Button type="submit" variant="primary" loading={loading} icon={UserPlus}>
                    Add Student
                  </Button>
                </div>
              </form>
            ) : (
              <div className="bulk-import-container">
                {!csvPreview ? (
                  <div 
                    className="upload-dropzone"
                    onClick={() => fileInputRef.current.click()}
                  >
                    <Upload size={48} className="upload-icon" />
                    <h4>Click or Drag CSV here</h4>
                    <p>Supported format: .csv</p>
                    <input 
                      type="file" 
                      ref={fileInputRef} 
                      style={{ display: 'none' }} 
                      accept=".csv"
                      onChange={handleFileUpload}
                    />
                  </div>
                ) : (
                  <div className="csv-preview">
                    <div className="preview-header">
                      <span><FileText size={16} /> {csvPreview.length} students found</span>
                      <button onClick={() => setCsvPreview(null)}>Change File</button>
                    </div>
                    <div className="preview-table-wrapper">
                      <table className="preview-table">
                        <thead>
                          <tr>
                            <th>Name</th>
                            <th>ID</th>
                            <th>Dept</th>
                          </tr>
                        </thead>
                        <tbody>
                          {csvPreview.slice(0, 5).map((s, i) => (
                            <tr key={i}>
                              <td>{s.name}</td>
                              <td>{s.enrollmentId}</td>
                              <td>{s.department}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {csvPreview.length > 5 && (
                        <div className="preview-more">And {csvPreview.length - 5} more...</div>
                      )}
                    </div>
                  </div>
                )}

                <div className="bulk-info">
                  <p><AlertCircle size={14} /> Make sure your CSV has headers: <strong>name, enrollmentId, age, gender, department, degreeGroup</strong></p>
                  <Button variant="ghost" size="sm" icon={Download}>Download Template</Button>
                </div>

                {error && <div className="error-message"><AlertCircle size={16} />{error}</div>}

                <div className="modal-footer">
                  <Button variant="ghost" onClick={handleClose}>Cancel</Button>
                  <Button 
                    variant="primary" 
                    disabled={!csvPreview} 
                    loading={loading} 
                    onClick={handleBulkSubmit}
                    icon={Upload}
                  >
                    Import Students
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </Modal>
  );
}
