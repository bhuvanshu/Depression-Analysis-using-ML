package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.dto.StudentResponse;
import com.bhuvanshu.mindcare.dto.StudentVerifyRequest;
import com.bhuvanshu.mindcare.entity.Student;
import com.bhuvanshu.mindcare.repository.StudentRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

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