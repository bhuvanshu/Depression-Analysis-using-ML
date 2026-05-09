# Use official Eclipse Temurin image for Java 17
FROM eclipse-temurin:17-jre-jammy

# Set the working directory
WORKDIR /app

# Copy the generated JAR file to the container
COPY target/*.jar app.jar

# Expose the port (Render sets PORT env variable)
EXPOSE 8080

# Run the jar
ENTRYPOINT ["java", "-jar", "app.jar"]
