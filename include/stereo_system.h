#ifndef STEREO_SYSTEM_H
#define STEREO_SYSTEM_H

#include <string>

/// \brief Main Lab program.
class StereoSystem
{
public:
    /// \brief Constructs the lab.
    /// \param data_path Optional path to parameter files.
    explicit StereoSystem(const std::string &data_path = "../param/");

    /// \brief Runs the lab.
    void run();

private:
    std::string data_path_;
    std::string window_name_;
};

#endif // STEREO_SYSTEM_H
