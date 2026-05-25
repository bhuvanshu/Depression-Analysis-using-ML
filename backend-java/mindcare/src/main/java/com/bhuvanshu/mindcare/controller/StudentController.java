package com.bhuvanshu.mindcare.controller;

import com.bhuvanshu.mindcare.dto.BulkAddResponse;
import com.bhuvanshu.mindcare.dto.StudentAddRequest;
import com.bhuvanshu.mindcare.dto.StudentResponse;
import com.bhuvanshu.mindcare.dto.StudentVerifyRequest;
import com.bhuvanshu.mindcare.service.StudentService;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/student")
public class StudentController {

    private static final Logger logger = LoggerFactory.getLogger(StudentController.class);

    @Autowired
    private StudentService studentService;

    @PostMapping("/verify")
    public ResponseEntity<?> verifyStudent(
            @RequestBody StudentVerifyRequest request) {

        StudentResponse response =
                studentService.verifyStudent(request);

        if (response == null) {
            return ResponseEntity
                    .badRequest()
                    .body("Student not found");
        }

        return ResponseEntity.ok(response);
    }

    @PostMapping("/add")
    public ResponseEntity<?> addStudent(
            @RequestBody StudentAddRequest request,
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {
        try {
            logger.info("POST /student/add | enrollmentId={}, college={}",
                    request.getEnrollmentId(), collegeName);
            StudentResponse response = studentService.addStudent(request, collegeName);
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        } catch (Exception e) {
            logger.error("Error adding student enrollmentId={}: {}", request.getEnrollmentId(), e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("An error occurred while adding the student: " + e.getMessage());
        }
    }

    @PostMapping("/bulk")
    public ResponseEntity<?> bulkAddStudents(
            @RequestBody List<StudentAddRequest> requests,
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {
        try {
            logger.info("POST /student/bulk | count={}, college={}", requests.size(), collegeName);
            BulkAddResponse response = studentService.bulkAddStudents(requests, collegeName);
            logger.info("Bulk upload result: uploaded={}, skipped={}", response.getTotalUploaded(), response.getSkippedDuplicates());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error during bulk upload for college '{}': {}", collegeName, e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("An error occurred during bulk upload: " + e.getMessage());
        }
    }
}