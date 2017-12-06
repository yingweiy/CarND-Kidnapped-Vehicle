/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#define EPSLON 1e-6

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 512;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i=0; i<num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    double x, y, theta;

    for (Particle& p: particles) {
        double theta1 = p.theta + yaw_rate * delta_t;
        if (fabs(yaw_rate)>EPSLON)
        {
            x = p.x + velocity/yaw_rate * (sin(theta1) - sin(p.theta));
            y = p.y + velocity/yaw_rate * (cos(p.theta) - cos(theta1));
        }
        else
        {
            x = p.x + velocity * cos(p.theta) * delta_t;
            y = p.y + velocity * sin(p.theta) * delta_t;
        }
        theta = theta1;

        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_theta(theta, std_pos[2]);

        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);

    }


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (LandmarkObs& obs: observations) {
        double min_dist = 1e6;
        for (LandmarkObs& pred: predicted) {
                double cur_dist = dist(pred.x, pred.y, obs.x, obs.y);
                if (cur_dist < min_dist) {
                    min_dist = cur_dist;
                    obs.id = pred.id;
                }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    // for each particle...
    for (Particle& p: particles) {

        // select predictions within sensor range
        vector<LandmarkObs> predictions;
        for (Map::single_landmark_s slm: map_landmarks.landmark_list) {
            if (dist(slm.x_f, slm.y_f, p.x, p.y) < sensor_range)
            {
                predictions.push_back(LandmarkObs{ slm.id_i, slm.x_f, slm.y_f });
            }
        }

        // convert the observations from vehicle to map coordinates
        vector<LandmarkObs> map_obs;
        for (const LandmarkObs& obs: observations) {
            double map_x = cos(p.theta)*obs.x - sin(p.theta)*obs.y + p.x;
            double map_y = sin(p.theta)*obs.x + cos(p.theta)*obs.y + p.y;
            map_obs.push_back(LandmarkObs{obs.id, map_x, map_y });
        }

        dataAssociation(predictions, map_obs);

        p.weight = 1.0;

        for (LandmarkObs& obs: map_obs) {

            // placeholders for observation and associated prediction coordinates
            double px, py;

            // get the x,y coordinates of the prediction associated with the current observation
            for (LandmarkObs& pred: predictions) {
                if (pred.id == obs.id) {
                    px = pred.x;
                    py = pred.y;
                    break;
                }
            }

            double obs_w = gaussian_prob(px, py, obs.x, obs.y, std_landmark[0], std_landmark[1]);

            // product of this obersvation weight with total observations weight
            p.weight *= obs_w;
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle> particles1;
    vector<double> weights;

    double max_w = 0.0;
    for (Particle& p: particles) {
        weights.push_back(p.weight);
        if (p.weight>max_w) {
            max_w = p.weight;
        }
    }

    uniform_int_distribution<int> uniform_i(0, num_particles-1);
    uniform_real_distribution<double> uniform_d(0.0, max_w);

    double beta = 0.0;
    int idx = uniform_i(gen);
    for (int i = 0; i < num_particles; i++) {
        beta += uniform_d(gen) * 2.0;
        while (beta > weights[idx]) {
            beta -= weights[idx];
            idx = (idx + 1) % num_particles;
        }
        particles1.push_back(particles[idx]);
    }

    particles = particles1;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
