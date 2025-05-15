# 自动 MOC / RCC / 包含生成目录
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)  # 启用自动RCC处理
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# ─── 可执行文件 ──────────────────────────────────────────────
add_executable(OpenMeshViewer
    viewer/OpenMeshViewer.cpp
    viewer/OpenMeshViewer.h
    ${CMAKE_SOURCE_DIR}/resources/resources.qrc
)

target_link_libraries(OpenMeshViewer
    PRIVATE
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        opengl32
        Eigen3::Eigen
    debug
        OpenMeshCored           OpenMeshToolsd
    optimized
        OpenMeshCore            OpenMeshTools
)

# ─── 运行时依赖复制 (Windows) ─────────────────────────────────
if(WIN32)
    # Qt DLL
    get_target_property(_qmake Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin "${_qmake}" DIRECTORY)
    set(DEBUG_D "$<$<CONFIG:Debug>:d>")

    add_custom_command(TARGET OpenMeshViewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_qt_bin}/Qt6Widgets${DEBUG_D}.dll"
            "${_qt_bin}/Qt6Core${DEBUG_D}.dll"
            "${_qt_bin}/Qt6Gui${DEBUG_D}.dll"
            "${_qt_bin}/Qt6OpenGL${DEBUG_D}.dll"
            "${_qt_bin}/Qt6OpenGLWidgets${DEBUG_D}.dll"
            $<TARGET_FILE_DIR:OpenMeshViewer>
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:OpenMeshViewer>/plugins/platforms"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_qt_bin}/../plugins/platforms/qwindows${DEBUG_D}.dll"
            "$<TARGET_FILE_DIR:OpenMeshViewer>/plugins/platforms"
    )

    # OpenMesh DLL
    add_custom_command(TARGET OpenMeshViewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENMESH_BIN}/OpenMeshCore${DEBUG_D}.dll"
            "${OPENMESH_BIN}/OpenMeshTools${DEBUG_D}.dll"
            $<TARGET_FILE_DIR:OpenMeshViewer>
    )
endif()
