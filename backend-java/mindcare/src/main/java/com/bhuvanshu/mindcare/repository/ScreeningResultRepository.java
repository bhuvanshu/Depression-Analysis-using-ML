package com.bhuvanshu.mindcare.repository;

import com.bhuvanshu.mindcare.entity.College;
import com.bhuvanshu.mindcare.entity.ScreeningResult;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ScreeningResultRepository
                extends JpaRepository<ScreeningResult, Long> {

        long countByRiskLevel(String riskLevel);

        List<ScreeningResult> findAllByOrderByPredictedAtDesc();

        List<ScreeningResult> findByRiskLevel(String riskLevel);

        // College-scoped queries

        @Query("SELECT sr FROM ScreeningResult sr " +
               "WHERE sr.screeningResponse.student.college = :college " +
               "ORDER BY sr.predictedAt DESC")
        List<ScreeningResult> findAllByCollegeOrderByPredictedAtDesc(
                        @Param("college") College college);

        @Query("SELECT sr FROM ScreeningResult sr " +
               "WHERE sr.screeningResponse.student.college = :college " +
               "AND sr.riskLevel = :riskLevel")
        List<ScreeningResult> findByCollegeAndRiskLevel(
                        @Param("college") College college,
                        @Param("riskLevel") String riskLevel);

        @Query("SELECT COUNT(sr) FROM ScreeningResult sr " +
               "WHERE sr.screeningResponse.student.college = :college " +
               "AND sr.riskLevel = :riskLevel")
        long countByCollegeAndRiskLevel(
                        @Param("college") College college,
                        @Param("riskLevel") String riskLevel);
}