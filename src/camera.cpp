#include <fmt/core.h>
#include <camera.h>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

void Camera::update(double deltaTime) {
    glm::mat4 cameraRotation = getRotationMatrix();
    position += (float)deltaTime * glm::vec3(cameraRotation * glm::vec4(velocity * 0.005f, 0.f));
}

void Camera::processSDLEvent(SDL_Event& e) {
    const Uint8* ks = SDL_GetKeyboardState(nullptr);

    int forward = (ks[SDL_SCANCODE_W] ? 1 : 0) - (ks[SDL_SCANCODE_S] ? 1 : 0);
    int right = (ks[SDL_SCANCODE_D] ? 1 : 0) - (ks[SDL_SCANCODE_A] ? 1 : 0);

    velocity.x = static_cast<float>(right);
    velocity.z = static_cast<float>(-forward);

    if (velocity.x != 0.f && velocity.z != 0.f) {
        const float inv = 0.70710678f; // 1/sqrt(2)
        velocity.x *= inv;
        velocity.z *= inv;
    }

    if (e.type == SDL_MOUSEMOTION) {
        yaw += (float)e.motion.xrel / 200.f;
        pitch -= (float)e.motion.yrel / 200.f;
    }
}

glm::mat4 Camera::getViewMatrix() {
    // to create a correct model view, we need to move the world in opposite
    // direction to the camera
    //  so we will create the camera model matrix and invert
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
    glm::mat4 cameraRotation = getRotationMatrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::getRotationMatrix() {
    // fairly typical FPS style camera. we join the pitch and yaw rotations into
    // the final rotation matrix

    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3{ 1.f, 0.f, 0.f });
    glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3{ 0.f, -1.f, 0.f });

    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}
