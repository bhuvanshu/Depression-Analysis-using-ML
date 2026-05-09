package com.bhuvanshu.mindcare.repository;

import com.bhuvanshu.mindcare.entity.ScreeningResult;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ScreeningResultRepository
                extends JpaRepository<ScreeningResult, Long> {

        long countByRiskLevel(String riskLevel);

        List<ScreeningResult> findAllByOrderByPredictedAtDesc();

        List<ScreeningResult> findByRiskLevel(String riskLevel);
}