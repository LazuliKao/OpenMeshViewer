#version 330 core

in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightPos = vec3(10.0, 10.0, 10.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform vec3 objectColor = vec3(0.3, 0.6, 0.9);

void main()
{
    // ¼òµ¥ Phong Âþ·´Éä¹âÕÕ
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);

    vec3 diffuse = diff * lightColor;
    vec3 result = diffuse * objectColor;
    FragColor = vec4(result, 1.0);
}
