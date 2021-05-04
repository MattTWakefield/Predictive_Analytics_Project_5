data(cells, package = "modeldata")
cells

set.seed(123)
cell_split<-initial_split(cells%>%select(-case), strata = class)
cell_train<-training(cell_split)
cell_test<-testing(cell_split)

tune_spec<-decision_tree(cost_complexity = tune(),
                         tree_depth = tune())%>%
  set_engine("rpart")%>%
  set_mode("classification")

tree_grid<-grid_regular(cost_complexity(),tree_depth(),levels = 5)
tree_grid

cell_folds<-vfold_cv(cell_train)

tree_wf<-workflow()%>%
  add_model(tune_spec)%>%
  add_formula(class ~ .)

tree_res<-tree_wf%>%tune_grid(resamples = cell_folds,
                              grid = tree_grid)

tree_res%>%collect_metrics()

best_tree<-tree_res%>%select_best("roc_auc")
final_wf<-tree_wf%>%finalize_workflow(best_tree)

final_tree<-final_wf%>%fit(data = cell_train)
final_fit<-final_wf%>%last_fit(cell_split)
final_fit%>%collect_metrics()
