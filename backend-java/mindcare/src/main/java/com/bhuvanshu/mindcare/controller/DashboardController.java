package com.bhuvanshu.mindcare.controller;

import com.bhuvanshu.mindcare.service.DashboardService;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/dashboard")
@CrossOrigin("*")
public class DashboardController {

    private static final Logger logger = LoggerFactory.getLogger(DashboardController.class);

    @Autowired
    private DashboardService dashboardService;

    // SUMMARY

    @GetMapping("/summary")
    public ResponseEntity<?> getSummary(
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {

        try {
            logger.info("GET /dashboard/summary | X-College-Name={}", collegeName);
            return ResponseEntity.ok(
                    dashboardService.getSummary(collegeName));
        } catch (Exception e) {
            logger.error("Error in GET /dashboard/summary for college '{}': {}", collegeName, e.getMessage(), e);
            return ResponseEntity.internalServerError()
                    .body("Error fetching dashboard summary: " + e.getMessage());
        }
    }

    // STUDENT TABLE

    @GetMapping("/students")
    public ResponseEntity<?> getStudents(
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {

        try {
            logger.info("GET /dashboard/students | X-College-Name={}", collegeName);
            return ResponseEntity.ok(
                    dashboardService.getAllStudents(collegeName));
        } catch (Exception e) {
            logger.error("Error in GET /dashboard/students for college '{}': {}", collegeName, e.getMessage(), e);
            return ResponseEntity.internalServerError()
                    .body("Error fetching students: " + e.getMessage());
        }
    }

    // CHART DATA

    @GetMapping("/charts")
    public ResponseEntity<?> getCharts(
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {

        try {
            logger.info("GET /dashboard/charts | X-College-Name={}", collegeName);
            return ResponseEntity.ok(
                    dashboardService
                            .getRiskDistributionChart(collegeName));
        } catch (Exception e) {
            logger.error("Error in GET /dashboard/charts for college '{}': {}", collegeName, e.getMessage(), e);
            return ResponseEntity.internalServerError()
                    .body("Error fetching chart data: " + e.getMessage());
        }
    }

    // HIGH RISK STUDENTS

    @GetMapping("/high-risk")
    public ResponseEntity<?> getHighRiskStudents(
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {

        try {
            logger.info("GET /dashboard/high-risk | X-College-Name={}", collegeName);
            return ResponseEntity.ok(
                    dashboardService
                            .getHighRiskStudents(collegeName));
        } catch (Exception e) {
            logger.error("Error in GET /dashboard/high-risk for college '{}': {}", collegeName, e.getMessage(), e);
            return ResponseEntity.internalServerError()
                    .body("Error fetching high-risk students: " + e.getMessage());
        }
    }
}