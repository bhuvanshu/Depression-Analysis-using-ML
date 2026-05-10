package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.dto.BulkAddResponse;
import com.bhuvanshu.mindcare.dto.StudentAddRequest;
import com.bhuvanshu.mindcare.dto.StudentResponse;
import com.bhuvanshu.mindcare.dto.StudentVerifyRequest;
import com.bhuvanshu.mindcare.entity.Student;
import com.bhuvanshu.mindcare.repository.StudentRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
public class StudentService {

    @Autowired
    private StudentRepository studentRepository;

    public StudentResponse verifyStudent(
            StudentVerifyRequest request) {

        Optional<Student> optionalStudent =
                studentRepository.findByEnrollmentId(
                        request.getEnrollmentId());

        if (optionalStudent.isEmpty()) {
            return null;
        }

        Student student = optionalStudent.get();
        return mapToResponse(student);
    }

    public StudentResponse addStudent(StudentAddRequest request) {
        if (request.getEnrollmentId() == null || request.getEnrollmentId().trim().isEmpty()) {
            throw new IllegalArgumentException("Enrollment ID cannot be empty");
        }

        if (studentRepository.existsByEnrollmentId(request.getEnrollmentId())) {
            throw new IllegalArgumentException("Student with this Enrollment ID already exists");
        }

        Student student = new Student();
        student.setEnrollmentId(request.getEnrollmentId());
        student.setName(request.getName());
        student.setAge(request.getAge());
        student.setGender(request.getGender());
        student.setDepartment(request.getDepartment());
        student.setDegreeGroup(request.getDegreeGroup());
        student.setCreatedAt(LocalDateTime.now());

        Student savedStudent = studentRepository.save(student);
        return mapToResponse(savedStudent);
    }

    public BulkAddResponse bulkAddStudents(List<StudentAddRequest> requests) {
        if (requests == null || requests.isEmpty()) {
            return new BulkAddResponse(0, 0, "No students provided for upload");
        }

        int uploaded = 0;
        int skipped = 0;
        List<Student> studentsToSave = new ArrayList<>();

        for (StudentAddRequest request : requests) {
            if (request.getEnrollmentId() == null || request.getEnrollmentId().trim().isEmpty()) {
                skipped++;
                continue;
            }

            if (studentRepository.existsByEnrollmentId(request.getEnrollmentId())) {
                skipped++;
                continue;
            }

            // Also check for duplicates within the current batch to avoid constraints violation
            boolean isDuplicateInBatch = studentsToSave.stream()
                    .anyMatch(s -> s.getEnrollmentId().equals(request.getEnrollmentId()));
            
            if (isDuplicateInBatch) {
                skipped++;
                continue;
            }

            Student student = new Student();
            student.setEnrollmentId(request.getEnrollmentId());
            student.setName(request.getName());
            student.setAge(request.getAge());
            student.setGender(request.getGender());
            student.setDepartment(request.getDepartment());
            student.setDegreeGroup(request.getDegreeGroup());
            student.setCreatedAt(LocalDateTime.now());
            
            studentsToSave.add(student);
        }

        if (!studentsToSave.isEmpty()) {
            studentRepository.saveAll(studentsToSave);
            uploaded = studentsToSave.size();
        }

        return new BulkAddResponse(uploaded, skipped, "Bulk upload completed successfully");
    }

    private StudentResponse mapToResponse(Student student) {
        StudentResponse response = new StudentResponse();
        response.setStudentId(student.getStudentId());
        response.setEnrollmentId(student.getEnrollmentId());
        response.setName(student.getName());
        response.setAge(student.getAge());
        response.setGender(student.getGender());
        response.setDepartment(student.getDepartment());
        response.setDegreeGroup(student.getDegreeGroup());
        return response;
    }
}