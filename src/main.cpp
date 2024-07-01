#include <iostream>
#include <string>
#include "../include/stereo_system.h"
#include "../include/disparity.h"

int main(int argc, char *argv[])
{
    /**
     * Usage: [-c | --camera <camera_id>]
     *        [-s | --size <width> <height>]
     *        [-d | --debug]
     *        [-h | --help]
     */
    const std::string param_path = "../calibration/";
    int camera_id = -1;  // -1 for file input
    int width = 1280;
    int height = 720;
    bool enable_debug = true;
    DisparityMapGenerator::DisparityMethod method = DisparityMapGenerator::SGM;

    if (argc > 1)
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "-c" || arg == "--camera")
            {
                if (i + 1 < argc)
                {
                    camera_id = std::stoi(argv[++i]);
                }
                else
                {
                    std::cerr << "Missing camera ID argument." << std::endl;
                    return EXIT_FAILURE;
                }
            }
            else if (arg == "-s" || arg == "--size")
            {
                if (i + 2 < argc)
                {
                    width = std::stoi(argv[++i]);
                    height = std::stoi(argv[++i]);
                }
                else
                {
                    std::cerr << "Missing width and/or height arguments." << std::endl;
                    return EXIT_FAILURE;
                }
            }
            else if (arg == "-d" || arg == "--debug")
            {
                enable_debug = true;
            }
            else if (arg == "-h" || arg == "--help")
            {
                std::cout << "Usage: [-c | --camera <camera_id>]" << std::endl
                          << "       [-s | --size <width> <height>]" << std::endl
                          << "       [-d | --debug]" << std::endl
                          << "       [-h | --help]" << std::endl;
                return EXIT_SUCCESS;
            }
        }
    }

    try
    {
        StereoSystem stereo_system(param_path, camera_id, width, height, method, enable_debug);
        stereo_system.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Caught exception:" << std::endl
                  << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception" << std::endl;
    }

    return EXIT_SUCCESS;
}
