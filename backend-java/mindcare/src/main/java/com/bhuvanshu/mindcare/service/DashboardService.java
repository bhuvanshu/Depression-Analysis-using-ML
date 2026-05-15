package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.dto.DashboardStudentResponse;
import com.bhuvanshu.mindcare.entity.College;
import com.bhuvanshu.mindcare.entity.ScreeningResponse;
import com.bhuvanshu.mindcare.entity.ScreeningResult;
import com.bhuvanshu.mindcare.entity.Student;
import com.bhuvanshu.mindcare.repository.CollegeRepository;
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

    @Autowired
    private CollegeRepository collegeRepository;

    // SUMMARY

    public Map<String, Object> getSummary(String collegeName) {

        Map<String, Object> summary = new HashMap<>();

        College college = resolveCollege(collegeName);

        if (college != null) {
            summary.put(
                    "totalStudents",
                    studentRepository.countByCollege(college));

            summary.put(
                    "highRisk",
                    screeningResultRepository
                            .countByCollegeAndRiskLevel(college, "High"));

            summary.put(
                    "moderateRisk",
                    screeningResultRepository
                            .countByCollegeAndRiskLevel(college, "Moderate"));

            summary.put(
                    "lowRisk",
                    screeningResultRepository
                            .countByCollegeAndRiskLevel(college, "Low"));
        } else {
            // Fallback: no college filter (backward compatibility)
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
        }

        return summary;
    }

    // STUDENT TABLE

    public List<DashboardStudentResponse> getAllStudents(String collegeName) {

        College college = resolveCollege(collegeName);

        List<ScreeningResult> results;

        if (college != null) {
            results = screeningResultRepository
                    .findAllByCollegeOrderByPredictedAtDesc(college);
        } else {
            results = screeningResultRepository
                    .findAllByOrderByPredictedAtDesc();
        }

        List<DashboardStudentResponse> responseList = new ArrayList<>();

        for (ScreeningResult result : results) {
            responseList.add(mapToDto(result));
        }

        return responseList;
    }

    // CHART DATA

    public List<Map<String, Object>> getRiskDistributionChart(String collegeName) {

        College college = resolveCollege(collegeName);

        List<Map<String, Object>> chartData = new ArrayList<>();

        Map<String, Object> high = new HashMap<>();
        high.put("label", "High");

        Map<String, Object> moderate = new HashMap<>();
        moderate.put("label", "Moderate");

        Map<String, Object> low = new HashMap<>();
        low.put("label", "Low");

        if (college != null) {
            high.put("count",
                    screeningResultRepository
                            .countByCollegeAndRiskLevel(college, "High"));

            moderate.put("count",
                    screeningResultRepository
                            .countByCollegeAndRiskLevel(college, "Moderate"));

            low.put("count",
                    screeningResultRepository
                            .countByCollegeAndRiskLevel(college, "Low"));
        } else {
            high.put("count",
                    screeningResultRepository
                            .countByRiskLevel("High"));

            moderate.put("count",
                    screeningResultRepository
                            .countByRiskLevel("Moderate"));

            low.put("count",
                    screeningResultRepository
                            .countByRiskLevel("Low"));
        }

        chartData.add(high);
        chartData.add(moderate);
        chartData.add(low);

        return chartData;
    }

    // HIGH RISK STUDENTS

    public List<DashboardStudentResponse> getHighRiskStudents(String collegeName) {

        College college = resolveCollege(collegeName);

        List<ScreeningResult> results;

        if (college != null) {
            results = screeningResultRepository
                    .findByCollegeAndRiskLevel(college, "High");
        } else {
            results = screeningResultRepository
                    .findByRiskLevel("High");
        }

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

    /**
     * Resolves a College entity from the collegeName.
     * Returns null if collegeName is blank (graceful fallback).
     */
    private College resolveCollege(String collegeName) {
        if (collegeName == null || collegeName.trim().isEmpty()) {
            return null;
        }
        return collegeRepository.findByCollegeName(collegeName).orElse(null);
    }
}