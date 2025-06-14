# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_A6gz-QawFQEXRuHn16DddgOnsorJRBD
"""

class LinearRegressionModel:
  def __init__(self):
    self.model = LinearRegression()

  def extract_features(self, X):
    # obtain the ego vehicle's data
    ego = X[:, 0, :, :] #(10000, 50, 6)

    # obtain ego vehicle positions, velocities, headings, and obj type
    ego_pos = ego[:, :, :2] #(10000, 50, 2)
    ego_vel = ego[:, :, 2:4] #(10000, 50, 2)
    ego_heading = ego[:, :, 4] #(10000, 50)
    ego_obj = ego[:, 0, 5].reshape(-1, 1) #(10000, 1)


    # timestamps = [0, 10, 20, 30, 40, -1]
    timestamps = np.arange(0, 50, 10, dtype = int).tolist() + [49]

    # obtain positions, velocities, and heading at every 5 timestamps, including final timestamp
    ego_pos_every_10 = ego_pos[:, timestamps, :].reshape(X.shape[0], -1) #(10000, 12)
    ego_vel_every_10 = ego_vel[:, timestamps, :].reshape(X.shape[0], -1) #(10000, 12)
    ego_heading_every_10 = ego_heading[:, timestamps].reshape(X.shape[0], -1) #(10000, 6)
    # ego_heading_every_10_sin = np.sin(ego_heading_every_10)
    # ego_heading_every_10_cos = np.cos(ego_heading_every_10)
    # ego_heading_every_10 = np.concatenate((ego_heading_every_10_sin, ego_heading_every_10_cos), axis = 1)

    # obtain pos, vel, and heading at last 5 timestamps
    timestamps_last_5 = np.arange(45, 50).tolist()
    ego_pos_last_10 = ego_pos[:, timestamps_last_5, :].reshape(X.shape[0], -1) #(10000, 10)
    ego_vel_last_10 = ego_vel[:, timestamps_last_5, :].reshape(X.shape[0], -1) #(10000, 10)
    ego_heading_last_10 = ego_heading[:, timestamps_last_5].reshape(X.shape[0], -1) #(10000, 5)
    # ego_heading_last_10_sin = np.sin(ego_heading_last_10)
    # ego_heading_last_10_cos = np.cos(ego_heading_last_10)
    # ego_heading_last_10 = np.concatenate((ego_heading_last_10_sin, ego_heading_last_10_cos), axis = 1)

    # calc average velocity and heading between timestamps (every 10 seconds)
    ego_vel_reshape = ego_vel.reshape(X.shape[0], 5, 10, 2)
    ego_avg_vel_every_10 = np.mean(ego_vel_reshape, axis = 2).reshape(X.shape[0], -1)
    ego_heading_reshape = ego_heading.reshape(X.shape[0], 5, 10)
    ego_avg_heading_every_10 = np.mean(ego_heading_reshape, axis = 2).reshape(X.shape[0], -1)
    # ego_avg_heading_every_10_sin = np.mean(np.sin(ego_heading_reshape), axis = 2)
    # ego_avg_heading_every_10_cos = np.mean(np.cos(ego_heading_reshape), axis = 2)
    # ego_avg_heading_every_10 = np.concatenate((ego_avg_heading_every_10_sin, ego_avg_heading_every_10_cos), axis = 1)


    # calc overall average velocity and heading
    ego_avg_vel = np.mean(ego_vel, axis = 1)
    ego_avg_heading = np.mean(ego_heading, axis = 1).reshape(-1, 1)
    # ego_avg_heading_sin = np.mean(np.sin(ego_heading), axis = 1).reshape(-1, 1)
    # ego_avg_heading_cos = np.mean(np.cos(ego_heading), axis = 1).reshape(-1, 1)
    # ego_avg_heading = np.concatenate((ego_avg_heading_sin, ego_avg_heading_cos), axis = 1)


    # calc acceleration at timestamps
    acc_5 = []
    for t in timestamps[1:]:
      acc = ego_vel[:, t, :] - ego_vel[:, t-1, :] #(10000, 2)
      acc_5.append(acc)
    ego_acc = np.concatenate(acc_5, axis=1)

    # calc distance from 5 closest agents at timestamps
    other_pos = X[:, 1:, :, :2]
    min_distances = []

    for t in timestamps:
      ego_pos_t = ego_pos[:, t, :] #(10000, 2)
      other_pos_t = other_pos[:, :, t, :] #(10000, 49, 2)

      distances = np.linalg.norm(ego_pos_t[:, np.newaxis, :] - other_pos_t, axis = -1) #(10000, 49)
      distances_sorted = np.sort(distances, axis = 1) #(10000, 49)
      min_distances_5 = distances_sorted[:, :5] #(10000, 5)
      min_distances.append(min_distances_5)

    ego_min_dist = np.concatenate(min_distances, axis = 1)

    # calc total distance traveled
    ego_pos_change = ego_pos[:, 1:, :] - ego_pos[:, :-1, :] #(10000, 49, 2)
    ego_distance_traveled = np.linalg.norm(ego_pos_change, axis = -1) #(10000, 49)
    ego_total_distance = np.sum(ego_distance_traveled, axis = 1).reshape(-1, 1) #(10000, 1)

    # features = np.concatenate((ego_pos_every_10, ego_pos_last_10, ego_min_dist, ego_total_distance,
    #                            ego_vel_every_10, ego_vel_last_10, ego_avg_vel, ego_avg_vel_every_10,
    #                            ego_acc,
    #                            ego_heading_every_10, ego_heading_last_10, ego_avg_heading, ego_avg_heading_every_10,
    #                            ego_obj), axis = 1)


    # features = np.concatenate((ego_pos_every_10, ego_pos_last_10,
    #                            ego_vel_every_10, ego_vel_last_10, ego_avg_vel,
    #                            ego_heading_every_10, ego_heading_last_10, ego_avg_heading), axis = 1)


    # features that obtained 13.08361 mse on kaggle
    # features = np.concatenate((ego_pos_every_10, ego_min_dist, ego_total_distance,
    #                            ego_vel_every_10, ego_avg_vel,
    #                            ego_heading_every_10, ego_avg_heading,
    #                            ego_obj), axis = 1)

    # features that obtained 13.00014 mse on kaggle
    features = np.concatenate((ego_pos_every_10,
                               ego_vel_every_10, ego_avg_vel,
                               ego_heading_every_10, ego_avg_heading,
                               ego_obj), axis = 1)
    return features

  # def fit(self, X, y):
  #   X_features = self.extract_features(X)
  #   y = y.reshape(y.shape[0], -1)
  #   self.model.fit(X_features, y)

  def fit(self, X, y, k=5):
    kf = KFold(n_splits=k)
    maes = []

    for step, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_features = self.extract_features(X_train)
        X_val_features = self.extract_features(X_val)

        self.model.fit(X_train_features, y_train.reshape(y_train.shape[0], -1))
        y_val_pred = self.model.predict(X_val_features).reshape(y_val.shape[0], 60, 2)

        mae = np.abs(y_val_pred - y_val).mean()
        maes.append(mae)
        print(f"Step {step+1}/{k} - MAE: {mae:.5f}")

    return maes

  def predict(self, X):
    X_features = self.extract_features(X)
    y_pred = self.model.predict(X_features)
    return y_pred.reshape(X.shape[0], 60, 2)

  def evaluate(self, X, y):
    y_pred = self.predict(X)
    mse = ((y_pred - y) ** 2).mean()
    print(mse)
    return mse

class RidgeRegressionModel:
  def __init__(self, alpha = 1.0):
    self.model = Ridge(alpha = alpha)


  def extract_features(self, X):
    # obtain the ego vehicle's data
    ego = X[:, 0, :, :] #(10000, 50, 6)

    # obtain ego vehicle positions, velocities, headings, and obj type
    ego_pos = ego[:, :, :2] #(10000, 50, 2)
    ego_vel = ego[:, :, 2:4] #(10000, 50, 2)
    ego_heading = ego[:, :, 4] #(10000, 50)
    ego_obj = ego[:, 0, 5].reshape(-1, 1) #(10000, 1)


    # timestamps = [0, 10, 20, 30, 40, -1]
    timestamps = np.arange(0, 50, 10, dtype = int).tolist() + [49]

    # obtain positions, velocities, and heading at every 5 timestamps, including final timestamp
    ego_pos_every_10 = ego_pos[:, timestamps, :].reshape(X.shape[0], -1) #(10000, 12)
    ego_vel_every_10 = ego_vel[:, timestamps, :].reshape(X.shape[0], -1) #(10000, 12)
    ego_heading_every_10 = ego_heading[:, timestamps].reshape(X.shape[0], -1) #(10000, 6)
    # ego_heading_every_10_sin = np.sin(ego_heading_every_10)
    # ego_heading_every_10_cos = np.cos(ego_heading_every_10)
    # ego_heading_every_10 = np.concatenate((ego_heading_every_10_sin, ego_heading_every_10_cos), axis = 1)

    # obtain pos, vel, and heading at last 5 timestamps
    timestamps_last_5 = np.arange(45, 50).tolist()
    ego_pos_last_10 = ego_pos[:, timestamps_last_5, :].reshape(X.shape[0], -1) #(10000, 10)
    ego_vel_last_10 = ego_vel[:, timestamps_last_5, :].reshape(X.shape[0], -1) #(10000, 10)
    ego_heading_last_10 = ego_heading[:, timestamps_last_5].reshape(X.shape[0], -1) #(10000, 5)
    # ego_heading_last_10_sin = np.sin(ego_heading_last_10)
    # ego_heading_last_10_cos = np.cos(ego_heading_last_10)
    # ego_heading_last_10 = np.concatenate((ego_heading_last_10_sin, ego_heading_last_10_cos), axis = 1)

    # calc average velocity and heading between timestamps (every 10 seconds)
    ego_vel_reshape = ego_vel.reshape(X.shape[0], 5, 10, 2)
    ego_avg_vel_every_10 = np.mean(ego_vel_reshape, axis = 2).reshape(X.shape[0], -1)
    ego_heading_reshape = ego_heading.reshape(X.shape[0], 5, 10)
    ego_avg_heading_every_10 = np.mean(ego_heading_reshape, axis = 2).reshape(X.shape[0], -1)
    # ego_avg_heading_every_10_sin = np.mean(np.sin(ego_heading_reshape), axis = 2)
    # ego_avg_heading_every_10_cos = np.mean(np.cos(ego_heading_reshape), axis = 2)
    # ego_avg_heading_every_10 = np.concatenate((ego_avg_heading_every_10_sin, ego_avg_heading_every_10_cos), axis = 1)


    # calc overall average velocity and heading
    ego_avg_vel = np.mean(ego_vel, axis = 1)
    ego_avg_heading = np.mean(ego_heading, axis = 1).reshape(-1, 1)
    # ego_avg_heading_sin = np.mean(np.sin(ego_heading), axis = 1).reshape(-1, 1)
    # ego_avg_heading_cos = np.mean(np.cos(ego_heading), axis = 1).reshape(-1, 1)
    # ego_avg_heading = np.concatenate((ego_avg_heading_sin, ego_avg_heading_cos), axis = 1)


    # calc acceleration at timestamps
    acc_5 = []
    for t in timestamps[1:]:
      acc = ego_vel[:, t, :] - ego_vel[:, t-1, :] #(10000, 2)
      acc_5.append(acc)
    ego_acc = np.concatenate(acc_5, axis=1)

    # calc distance from 5 closest agents at timestamps
    other_pos = X[:, 1:, :, :2]
    min_distances = []

    for t in timestamps:
      ego_pos_t = ego_pos[:, t, :] #(10000, 2)
      other_pos_t = other_pos[:, :, t, :] #(10000, 49, 2)

      distances = np.linalg.norm(ego_pos_t[:, np.newaxis, :] - other_pos_t, axis = -1) #(10000, 49)
      distances_sorted = np.sort(distances, axis = 1) #(10000, 49)
      min_distances_5 = distances_sorted[:, :5] #(10000, 5)
      min_distances.append(min_distances_5)

    ego_min_dist = np.concatenate(min_distances, axis = 1)

    # calc total distance traveled
    ego_pos_change = ego_pos[:, 1:, :] - ego_pos[:, :-1, :] #(10000, 49, 2)
    ego_distance_traveled = np.linalg.norm(ego_pos_change, axis = -1) #(10000, 49)
    ego_total_distance = np.sum(ego_distance_traveled, axis = 1).reshape(-1, 1) #(10000, 1)

    # obtained 14.30566 mse on kaggle
    # features = np.concatenate((ego_pos_every_10, ego_pos_last_10, ego_min_dist, ego_total_distance,
    #                            ego_vel_every_10, ego_vel_last_10, ego_avg_vel, ego_avg_vel_every_10,
    #                            ego_acc,
    #                            ego_heading_every_10, ego_heading_last_10, ego_avg_heading, ego_avg_heading_every_10,
    #                            ego_obj), axis = 1)


    # features = np.concatenate((ego_pos_every_10, ego_pos_last_10,
    #                            ego_vel_every_10, ego_vel_last_10, ego_avg_vel,
    #                            ego_heading_every_10, ego_heading_last_10, ego_avg_heading), axis = 1)


    # features that obtained 11.97326 mse on kaggle
    # default alpha: 11.97326 mse
    features = np.concatenate((ego_pos_every_10,
                               ego_vel_every_10, ego_avg_vel,
                               ego_heading_every_10, ego_avg_heading,
                               ego_obj), axis = 1)
    return features

  # def fit(self, X, y):
  #   X_features = self.extract_features(X)
  #   y = y.reshape(y.shape[0], -1)
  #   self.model.fit(X_features, y)

  def fit(self, X, y, k=5):
    kf = KFold(n_splits=k)
    maes = []

    for step, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_features = self.extract_features(X_train)
        X_val_features = self.extract_features(X_val)

        self.model.fit(X_train_features, y_train.reshape(y_train.shape[0], -1))
        y_val_pred = self.model.predict(X_val_features).reshape(y_val.shape[0], 60, 2)

        mae = np.abs(y_val_pred - y_val).mean()
        maes.append(mae)
        print(f"Step {step+1}/{k} - MAE: {mae:.5f}")

    return maes

  def predict(self, X):
    X_features = self.extract_features(X)
    y_pred = self.model.predict(X_features)
    return y_pred.reshape(X.shape[0], 60, 2)

  def evaluate(self, X, y):
    y_pred = self.predict(X)
    mse = ((y_pred - y) ** 2).mean()
    print(mse)
    return mse

class LassoRegressionModel:
  def __init__(self, alpha = 0.1):
    self.model = Lasso(alpha = alpha)


  def extract_features(self, X):
    # obtain the ego vehicle's data
    ego = X[:, 0, :, :] #(10000, 50, 6)

    # obtain ego vehicle positions, velocities, headings, and obj type
    ego_pos = ego[:, :, :2] #(10000, 50, 2)
    ego_vel = ego[:, :, 2:4] #(10000, 50, 2)
    ego_heading = ego[:, :, 4] #(10000, 50)
    ego_obj = ego[:, 0, 5].reshape(-1, 1) #(10000, 1)


    # timestamps = [0, 10, 20, 30, 40, -1]
    timestamps = np.arange(0, 50, 10, dtype = int).tolist() + [49]

    # obtain positions, velocities, and heading at every 5 timestamps, including final timestamp
    ego_pos_every_10 = ego_pos[:, timestamps, :].reshape(X.shape[0], -1) #(10000, 12)
    ego_vel_every_10 = ego_vel[:, timestamps, :].reshape(X.shape[0], -1) #(10000, 12)
    ego_heading_every_10 = ego_heading[:, timestamps].reshape(X.shape[0], -1) #(10000, 6)
    # ego_heading_every_10_sin = np.sin(ego_heading_every_10)
    # ego_heading_every_10_cos = np.cos(ego_heading_every_10)
    # ego_heading_every_10 = np.concatenate((ego_heading_every_10_sin, ego_heading_every_10_cos), axis = 1)

    # obtain pos, vel, and heading at last 5 timestamps
    timestamps_last_5 = np.arange(45, 50).tolist()
    ego_pos_last_10 = ego_pos[:, timestamps_last_5, :].reshape(X.shape[0], -1) #(10000, 10)
    ego_vel_last_10 = ego_vel[:, timestamps_last_5, :].reshape(X.shape[0], -1) #(10000, 10)
    ego_heading_last_10 = ego_heading[:, timestamps_last_5].reshape(X.shape[0], -1) #(10000, 5)
    # ego_heading_last_10_sin = np.sin(ego_heading_last_10)
    # ego_heading_last_10_cos = np.cos(ego_heading_last_10)
    # ego_heading_last_10 = np.concatenate((ego_heading_last_10_sin, ego_heading_last_10_cos), axis = 1)

    # calc average velocity and heading between timestamps (every 10 seconds)
    ego_vel_reshape = ego_vel.reshape(X.shape[0], 5, 10, 2)
    ego_avg_vel_every_10 = np.mean(ego_vel_reshape, axis = 2).reshape(X.shape[0], -1)
    ego_heading_reshape = ego_heading.reshape(X.shape[0], 5, 10)
    ego_avg_heading_every_10 = np.mean(ego_heading_reshape, axis = 2).reshape(X.shape[0], -1)
    # ego_avg_heading_every_10_sin = np.mean(np.sin(ego_heading_reshape), axis = 2)
    # ego_avg_heading_every_10_cos = np.mean(np.cos(ego_heading_reshape), axis = 2)
    # ego_avg_heading_every_10 = np.concatenate((ego_avg_heading_every_10_sin, ego_avg_heading_every_10_cos), axis = 1)


    # calc overall average velocity and heading
    ego_avg_vel = np.mean(ego_vel, axis = 1)
    ego_avg_heading = np.mean(ego_heading, axis = 1).reshape(-1, 1)
    # ego_avg_heading_sin = np.mean(np.sin(ego_heading), axis = 1).reshape(-1, 1)
    # ego_avg_heading_cos = np.mean(np.cos(ego_heading), axis = 1).reshape(-1, 1)
    # ego_avg_heading = np.concatenate((ego_avg_heading_sin, ego_avg_heading_cos), axis = 1)


    # calc acceleration at timestamps
    acc_5 = []
    for t in timestamps[1:]:
      acc = ego_vel[:, t, :] - ego_vel[:, t-1, :] #(10000, 2)
      acc_5.append(acc)
    ego_acc = np.concatenate(acc_5, axis=1)

    # calc distance from 5 closest agents at timestamps
    other_pos = X[:, 1:, :, :2]
    min_distances = []

    for t in timestamps:
      ego_pos_t = ego_pos[:, t, :] #(10000, 2)
      other_pos_t = other_pos[:, :, t, :] #(10000, 49, 2)

      distances = np.linalg.norm(ego_pos_t[:, np.newaxis, :] - other_pos_t, axis = -1) #(10000, 49)
      distances_sorted = np.sort(distances, axis = 1) #(10000, 49)
      min_distances_5 = distances_sorted[:, :5] #(10000, 5)
      min_distances.append(min_distances_5)

    ego_min_dist = np.concatenate(min_distances, axis = 1)

    # calc total distance traveled
    ego_pos_change = ego_pos[:, 1:, :] - ego_pos[:, :-1, :] #(10000, 49, 2)
    ego_distance_traveled = np.linalg.norm(ego_pos_change, axis = -1) #(10000, 49)
    ego_total_distance = np.sum(ego_distance_traveled, axis = 1).reshape(-1, 1) #(10000, 1)

    # obtained 11.82991 on kaggle
    # default alpha: 11.82991 mse
    features = np.concatenate((ego_pos_every_10, ego_pos_last_10, ego_min_dist, ego_total_distance,
                               ego_vel_every_10, ego_vel_last_10, ego_avg_vel, ego_avg_vel_every_10,
                               ego_acc,
                               ego_heading_every_10, ego_heading_last_10, ego_avg_heading, ego_avg_heading_every_10,
                               ego_obj), axis = 1)


    # features = np.concatenate((ego_pos_every_10, ego_pos_last_10,
    #                            ego_vel_every_10, ego_vel_last_10, ego_avg_vel,
    #                            ego_heading_every_10, ego_heading_last_10, ego_avg_heading), axis = 1)


    # features that obtained 13.08361 mse on kaggle
    # features = np.concatenate((ego_pos_every_10, ego_min_dist, ego_total_distance,
    #                            ego_vel_every_10, ego_avg_vel,
    #                            ego_heading_every_10, ego_avg_heading,
    #                            ego_obj), axis = 1)

    # features that obtained 13.00014 mse on kaggle
    # features = np.concatenate((ego_pos_every_10,
    #                            ego_vel_every_10, ego_avg_vel,
    #                            ego_heading_every_10, ego_avg_heading,
    #                            ego_obj), axis = 1)
    return features

  def fit(self, X, y, k=5):
    kf = KFold(n_splits=k)
    maes = []

    for step, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train_features = self.extract_features(X_train)
        X_val_features = self.extract_features(X_val)

        self.model.fit(X_train_features, y_train.reshape(y_train.shape[0], -1))
        y_val_pred = self.model.predict(X_val_features).reshape(y_val.shape[0], 60, 2)

        mae = np.abs(y_val_pred - y_val).mean()
        maes.append(mae)
        print(f"Step {step+1}/{k} - MAE: {mae:.5f}")
    return maes

  def predict(self, X):
    X_features = self.extract_features(X)
    y_pred = self.model.predict(X_features)
    return y_pred.reshape(X.shape[0], 60, 2)

  def evaluate(self, X, y):
    y_pred = self.predict(X)
    mse = ((y_pred - y) ** 2).mean()
    print(mse)
    return mse

## Example training:

# # initialize and train
# Ridge_model = RidgeRegressionModel()
# maes = Ridge_model.fit(train_x, train_y) # this line returns maes of each k split
# Ridge_model.evaluate(train_x, train_y)

# # predict test
# test_x = test_data[..., :50, :]
# ridge_pred_y = Ridge_model.predict(test_x) #(2100, 60, 2)

# # save to file
# output_df = pd.DataFrame(ridge_pred_y.reshape(-1, 2), columns=["x", "y"])
# output_df.index.name = "index"

# output_df.to_csv("RR_baseline.csv")