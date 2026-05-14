package com.bhuvanshu.mindcare;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import jakarta.annotation.PostConstruct;
import java.util.TimeZone;

@SpringBootApplication
public class MindcareApplication {

	@PostConstruct
	public void init() {
		// Set JVM timezone to IST to fix deployment timestamp discrepancies
		TimeZone.setDefault(TimeZone.getTimeZone("Asia/Kolkata"));
	}

	public static void main(String[] args) {
		SpringApplication.run(MindcareApplication.class, args);
	}

}
