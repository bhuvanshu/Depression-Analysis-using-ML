package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.dto.DashboardStudentResponse;
import com.bhuvanshu.mindcare.entity.ScreeningResponse;
import com.bhuvanshu.mindcare.entity.ScreeningResult;
import com.bhuvanshu.mindcare.entity.Student;
import com.bhuvanshu.mindcare.repository.ScreeningResultRepository;
import com.bhuvanshu.mindcare.repository.StudentRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class DashboardService {

    @Autowired
    private StudentRepository studentRepository;

    @Autowired
    private ScreeningResultRepository screeningResultRepository;

    // SUMMARY

    public Map<String, Object> getSummary() {

        Map<String, Object> summary = new HashMap<>();

        summary.put(
                "totalStudents",
                studentRepository.count());

        summary.put(
                "highRisk",
                screeningResultRepository
                        .countByRiskLevel("High"));

        summary.put(
                "moderateRisk",
                screeningResultRepository
                        .countByRiskLevel("Moderate"));

        summary.put(
                "lowRisk",
                screeningResultRepository
                        .countByRiskLevel("Low"));

        return summary;
    }

    // STUDENT TABLE

    public List<DashboardStudentResponse> getAllStudents() {

        List<ScreeningResult> results = screeningResultRepository
                .findAllByOrderByPredictedAtDesc();

        List<DashboardStudentResponse> responseList = new ArrayList<>();

        for (ScreeningResult result : results) {
            responseList.add(mapToDto(result));
        }

        return responseList;
    }

    // CHART DATA

    public List<Map<String, Object>> getRiskDistributionChart() {

        List<Map<String, Object>> chartData = new ArrayList<>();

        Map<String, Object> high = new HashMap<>();

        high.put("label", "High");

        high.put(
                "count",
                screeningResultRepository
                        .countByRiskLevel("High"));

        chartData.add(high);

        Map<String, Object> moderate = new HashMap<>();

        moderate.put("label", "Moderate");

        moderate.put(
                "count",
                screeningResultRepository
                        .countByRiskLevel("Moderate"));

        chartData.add(moderate);

        Map<String, Object> low = new HashMap<>();

        low.put("label", "Low");

        low.put(
                "count",
                screeningResultRepository
                        .countByRiskLevel("Low"));

        chartData.add(low);

        return chartData;
    }

    // HIGH RISK STUDENTS

    public List<DashboardStudentResponse> getHighRiskStudents() {

        List<ScreeningResult> results = screeningResultRepository
                .findByRiskLevel("High");

        List<DashboardStudentResponse> responseList = new ArrayList<>();

        for (ScreeningResult result : results) {
            responseList.add(mapToDto(result));
        }

        return responseList;
    }

    // SHARED MAPPING HELPER

    private DashboardStudentResponse mapToDto(ScreeningResult result) {

        ScreeningResponse screening = result.getScreeningResponse();
        Student student = screening.getStudent();

        DashboardStudentResponse dto = new DashboardStudentResponse();

        // Identity fields
        dto.setEnrollmentId(student.getEnrollmentId());
        dto.setStudentName(student.getName());
        dto.setDepartment(student.getDepartment());

        // Risk fields
        dto.setRiskLevel(result.getRiskLevel());
        dto.setProbabilityScore(result.getProbabilityScore());

        // Questionnaire metrics from ScreeningResponse
        dto.setAcademicPressure(screening.getAcademicPressure());
        dto.setFinancialStress(screening.getFinancialStress());
        dto.setStudySatisfaction(screening.getStudySatisfaction());
        dto.setSuicidalThoughts(screening.getSuicidalThoughts());
        dto.setFamilyHistory(screening.getFamilyHistory());
        dto.setStudyHours(screening.getWorkStudyHours());
        dto.setSleepDuration(screening.getSleepDuration());

        // Screening date from prediction timestamp
        dto.setScreeningDate(result.getPredictedAt());

        return dto;
    }
}