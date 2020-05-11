import os

import constants



class Core(object):
    def get_args(self):
        # to be implemented in sub-class
        return None


    def get_trainer(self):
        # to be implemented in sub-class
        return None


    def run(self):
        ###############################
        # Make necessary directories

        if not os.path.exists(constants.LOG_DIR):
            os.mkdir(constants.LOG_DIR)
        if not os.path.exists(constants.MODEL_DIR):
            os.mkdir(constants.MODEL_DIR)


        ###############################
        # Get arguments and initialize extractor

        args = self.get_args()
        trainer = self.get_trainer(args)

        ###############################
        # Setup

        trainer.init_hyperparameters()
        if args.train_data and args.valid_data:
            trainer.load_training_and_validation_data()
        if args.test_data:
            trainer.load_test_data()

        ###############################
        # Run

        trainer.run()

        ###############################
        # Summary

        trainer.summary()

        ###############################
        # Terminate

        trainer.close()
