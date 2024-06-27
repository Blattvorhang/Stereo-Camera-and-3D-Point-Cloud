#include <iostream>
#include "../include/stereo_system.h"

int main()
{
    try
    {
        StereoSystem stereo_system;
        stereo_system.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Caught exception:\n"
                  << e.what() << "\n";
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_SUCCESS;
}
