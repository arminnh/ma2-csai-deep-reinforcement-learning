// 1 hidden layer, no temporal window, large view around the car

//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
// Defines the most basic settings – for larger inputs you should probably increase the number of train iterations. Actually looking ahead a few patches, and at least one lane to the side is probably a good idea as well.
lanesSide = 3;
patchesAhead = 30;
patchesBehind = 10;
trainIterations = 10000;

// Specifies some more details about the input – you don’t need to touch this part (except maybe the temporal window).
var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 0;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

var layer_defs = [];
// The net is defined with an array of layers starting with the input which you don’t have to change:
layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
// We added one basic hidden layer with just one neuron to show you how to do that – you should definitely change that
layer_defs.push({
    type: 'fc',
    num_neurons: 75,
    activation: 'relu'
});
// And in the end there is the final (L2) regression layer that decides on the action, which probably is fine as it is
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

// options for the Temporal Difference learner that trains the above net
// by backpropping the temporal difference learning rule.
var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 64,
    l2_decay: 0.01
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 3000;
opt.start_learn_threshold = 500;
opt.gamma = 0.7;
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.05;
opt.epsilon_test_time = 0.2;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;
// There are a lot more options for the Q-Learning part – details on them can be found in the comments of the code at the following link: https://github.com/karpathy/convnetjs/blob/master/build/deepqlearn.js These are mostly interesting for more advanced optimisations of your net.

// And the last step is creating the brain.
brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

// no action = 0, accelerate = 1, decelerate = 2, goLeft = 3, goRight = 4
learn = function (state, lastReward) {
   // communicate the reward
   brain.backward(lastReward);

    // get optimal action from learned policy
    var action = brain.forward(state);

    draw_net();
    draw_stats();

    return action;
}

//]]>
