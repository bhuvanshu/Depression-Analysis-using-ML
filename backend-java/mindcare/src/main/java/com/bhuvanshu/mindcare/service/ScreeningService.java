package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.dto.ScreeningRequest;
import com.bhuvanshu.mindcare.dto.ScreeningResultResponse;
import com.bhuvanshu.mindcare.entity.ScreeningResponse;
import com.bhuvanshu.mindcare.entity.ScreeningResult;
import com.bhuvanshu.mindcare.entity.Student;
import com.bhuvanshu.mindcare.repository.ScreeningResponseRepository;
import com.bhuvanshu.mindcare.repository.ScreeningResultRepository;
import com.bhuvanshu.mindcare.repository.StudentRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class ScreeningService {

        @Autowired
        private RestTemplate restTemplate;

        @Autowired
        private StudentRepository studentRepository;

        @Autowired
        private ScreeningResponseRepository screeningResponseRepository;

        @Autowired
        private ScreeningResultRepository screeningResultRepository;

        private final String ML_API = "http://localhost:5000/predict";

        public ScreeningResultResponse submitScreening(
                        ScreeningRequest request) {

                // FETCH STUDENT

                Student student = studentRepository
                                .findByEnrollmentId(
                                                request.getEnrollmentId())
                                .orElseThrow(
                                                () -> new RuntimeException(
                                                                "Student not found"));

                // SAVE QUESTIONNAIRE RESPONSE

                ScreeningResponse responseEntity = new ScreeningResponse();

                responseEntity.setStudent(student);

                responseEntity.setAcademicPressure(
                                request.getAcademic_pressure());

                responseEntity.setFinancialStress(
                                request.getFinancial_stress());

                responseEntity.setStudySatisfaction(
                                request.getStudy_satisfaction());

                responseEntity.setWorkStudyHours(
                                request.getWork_study_hours());

                responseEntity.setCgpa(
                                request.getCgpa());

                responseEntity.setFamilyHistory(
                                request.getFamily_history());

                responseEntity.setSuicidalThoughts(
                                request.getSuicidal_thoughts());

                screeningResponseRepository.save(
                                responseEntity);

                // CALL ML API

                HttpHeaders headers = new HttpHeaders();

                headers.setContentType(
                                MediaType.APPLICATION_JSON);

                HttpEntity<ScreeningRequest> entity = new HttpEntity<>(request, headers);

                ResponseEntity<ScreeningResultResponse> mlResponse = restTemplate.postForEntity(
                                ML_API,
                                entity,
                                ScreeningResultResponse.class);

                ScreeningResultResponse prediction = mlResponse.getBody();

                // SAVE PREDICTION RESULT

                ScreeningResult result = new ScreeningResult();

                result.setScreeningResponse(
                                responseEntity);

                result.setRiskLevel(
                                prediction.getRisk_level());

                result.setRecommendation(
                                prediction.getRecommended_action());

                result.setProbabilityScore(
                                prediction.getProbability()
                                                .getDepressed());

                screeningResultRepository.save(result);

                return prediction;
        }
}